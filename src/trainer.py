import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup
import wandb
import functools
from nltk import PorterStemmer
from rouge import Rouge
from spacy.lang.en import English
from tabulate import tabulate

#kaggle
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()

from pathlib import Path
from tqdm import tqdm
import pandas as pd
from time import time
import math
import os

from src.data import ELI5DatasetS2S, load_data, format_docs

def prepare_dataset(logger, args, data_type="train"):
    """
    Load dataset from disk

    :param
        logger: logging
                log information
        args: Arguments
                contains hyper-parameters
        data_type: str
                specify training or evaluation dataset

    :return
        dataset: src.data.Dataset
                custome Dataset object
    """
    if args.logger:
        logger.info('Creating Dataset...')
    elif args.is_notebook:
        print('Creating Dataset...')

    examples = load_data(args.train_data if data_type == "train" else args.eval_data)
    example_docs = format_docs(examples)
    dataset = ELI5DatasetS2S(examples, document_cache=example_docs)

    if args.logger:
        logger.info('Creating Dataset is done.')
    elif args.is_notebook:
        print('Creating Dataset is done.')

    return dataset

def prepare_training_stuff(logger, args):
    if args.logger:
        logger.info('Creating model and tokenizer...')
    elif args.is_notebook:
        print('Creating model and tokenizer...')

    full_model_name = args.model_name + '-' + args.model_size 
    
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(full_model_name).to(args.device)
    if args.model_path is not None:
        param_dict = torch.load(args.model_path)  # has model weights, optimizer, and scheduler states
        model.load_state_dict(param_dict["model"])

    if args.logger:
        logger.info('Creating model and tokenizer is done')
    elif args.is_notebook:
        print('Creating model and tokenizer is done')

    return tokenizer, model

class Trainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, args, logger):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.args = args
        self.logger = logger
        self.optimizer = None
        self.scheduler = None
        self.model_save_name = self.args.model_name +  '-' + self.args.model_size

        if self.args.is_main:
            try:
                wandb_api = user_secrets.get_secret("wandb_api") 
                wandb.login(key=wandb_api)
                self.wandb_logger = True
                wandb.init(
                    project='ans-gen-BART',
                    name='experiment-1')
            except:
                self.wandb_logger = False
                if self.args.logger:
                    self.logger.warning("Wandb is not available.")
        else:
            self.wandb_logger = False

        torch.manual_seed(self.args.seed)

    def make_qa_s2s_batch(self, qa_list, tokenizer, max_len=64, max_a_len=256, device="cuda:0"):
        q_ls = [q for q, a in qa_list]
        a_ls = [a for q, a in qa_list]
        q_toks = tokenizer.batch_encode_plus(q_ls, max_length=max_len, pad_to_max_length=True)
        q_ids, q_mask = (
            torch.LongTensor(q_toks["input_ids"]).to(device),
            torch.LongTensor(q_toks["attention_mask"]).to(device),
        )
        a_toks = tokenizer.batch_encode_plus(a_ls, max_length=max_a_len, pad_to_max_length=True)
        a_ids, a_mask = (
            torch.LongTensor(a_toks["input_ids"]).to(device),
            torch.LongTensor(a_toks["attention_mask"]).to(device),
        )
        labels = a_ids[:, 1:].contiguous().clone()
        labels[a_mask[:, 1:].contiguous() == 0] = -100
        model_inputs = {
            "input_ids": q_ids,
            "attention_mask": q_mask,
            "decoder_input_ids": a_ids[:, :-1].contiguous(),
            "labels": labels}   
        return model_inputs

    def train_qa_s2s_epoch(self, e=0, curriculum=False):
        self.model.train()
        # make iterator 

        if curriculum:
            train_sampler = SequentialSampler(self.train_dataset)
        else:
            train_sampler = RandomSampler(self.train_dataset)

        model_collate_fn = functools.partial(
            self.make_qa_s2s_batch, tokenizer=self.tokenizer, max_len=self.args.max_input_length, max_a_len=self.args.max_ans_length, device=self.args.device
        )
        data_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
        epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
        # accumulate loss since last print
        loc_steps = 0
        loc_loss = 0.0
        st_time = time()

        save_epoch_path = os.path.join(self.args.checkpoint_path, 'epoch{}'.format(e))
        if not os.path.exists(save_epoch_path):
            os.makedirs(save_epoch_path)
        for step, batch_inputs in enumerate(epoch_iterator):
            outputs = self.model(**batch_inputs)
            loss = outputs.loss
            if self.wandb_logger:
                wandb.log({"train_loss": loss})
            loss.backward()

            # optimizer
            if step % self.args.backward_freq == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

            # some printing within the epoch
            loc_loss += loss.item()
            loc_steps += 1
            if step % self.args.train_print_freq == 0 or step == math.floor(len(self.train_dataset) / self.args.batch_size) or step == 1:
                if self.args.logger:
                    self.logger.info(
                        "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                            e, step, len(self.train_dataset) // self.args.batch_size, loc_loss / loc_steps, time() - st_time,
                        )
                    )
                elif self.args.is_notebook:
                    print(
                        "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                            e, step, len(self.train_dataset) // self.args.batch_size, loc_loss / loc_steps, time() - st_time,
                        )
                    )
                loc_loss = 0
                loc_steps = 0
            if step != 0 and (step % self.args.save_freq == 0 or step == math.floor(len(self.train_dataset) / self.args.batch_size)):
                if self.args.logger:
                    self.logger.info("Saving model {}_{}_{}".format(self.model_save_name, e, step))
                elif self.args.is_notebook:
                    print("Saving model {}_{}_{}".format(self.model_save_name, e, step))

                m_save_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict()}
                torch.save(m_save_dict, os.path.join(save_epoch_path, "{}_{}_{}.pth".format(self.model_save_name.replace("facebook/",""), e, step)))

    def eval_qa_s2s_epoch(self):
        self.model.eval()
        # make iterator
        train_sampler = SequentialSampler(self.eval_dataset)
        model_collate_fn = functools.partial(
            self.make_qa_s2s_batch, tokenizer=self.tokenizer, max_len=self.args.max_input_length, max_a_len=self.args.max_ans_length, device=self.args.device)

        data_loader = DataLoader(self.eval_dataset, batch_size=self.args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
        epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
        # accumulate loss since last print
        loc_steps = 0
        loc_loss = 0.0
        st_time = time()
        with torch.no_grad():
            for step, batch_inputs in enumerate(epoch_iterator):
                outputs = self.model(**batch_inputs)
                loss = outputs.loss
                if self.wandb_logger:
                  wandb.log({"val_loss": loss})

                # some printing within the epoch
                
                loc_loss += loss.item()
                loc_steps += 1
                
                if step % self.args.eval_print_freq == 0 or step == math.floor(len(self.eval_dataset) / self.args.batch_size):
                    if self.args.logger:
                        self.logger.info("{:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(step, len(self.eval_dataset) // self.args.batch_size, loc_loss / loc_steps, time() - st_time))
                    elif self.args.is_notebook:
                        print("{:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(step, len(self.eval_dataset) // self.args.batch_size, loc_loss / loc_steps, time() - st_time))
        if self.args.logger:
            self.logger.info("Total \t L: {:.3f} \t -- {:.3f}".format(loc_loss / loc_steps, time() - st_time,))
        elif self.args.is_notebook:
            print("Total \t L: {:.3f} \t -- {:.3f}".format(loc_loss / loc_steps, time() - st_time,))

    def train(self):
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=400,
            num_training_steps=(self.args.num_epochs + 1) * math.ceil(len(self.train_dataset) / self.args.batch_size))

        if (self.args.model_path is not None) and (self.args.retrain == False):
            if self.args.logger:
                self.logger.info('Skip training model. Evaluating...')
            elif self.args.is_notebook:
                print('Skip training model. Evaluating...')

            self.eval_qa_s2s_epoch()
        else:
            for e in range(self.args.num_epochs):
                self.train_qa_s2s_epoch(
                    e,
                    curriculum=(e == 0))

                m_save_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                }

                if self.args.logger:
                    self.logger.info("Evaluate after {} epoch(s)".format(e))
                elif self.args.is_notebook:
                    print("Evaluate after {} epoch(s)".format(e))

                self.eval_qa_s2s_epoch()

                if self.args.logger:
                    self.logger.info("Saving model {}_{} after {} epoch(s)".format(self.model_save_name, e, e))
                elif self.args.is_notebook:
                    print("Saving model {}_{}".format(self.model_save_name, e))
                torch.save(m_save_dict, os.path.join(self.args.checkpoint_path, "{}_{}.pth".format(self.model_save_name.replace("facebook/",""), e)))
        if self.wandb_logger:
            wandb.finish()

    # generate answer from input "question: ... context: <p> ..."
    def generate(
        self,
        question_doc,
        model,
        tokenizer,
        num_answers=1,
        num_beams=None,
        min_len=64,
        max_len=256,
        do_sample=False,
        temp=1.0,
        top_p=None,
        top_k=None,
        max_input_length=512,
        device="cuda:0",
    ):
        model_inputs = self.make_qa_s2s_batch([(question_doc, "A")], tokenizer, max_input_length, device=device)
        n_beams = num_answers if num_beams is None else max(num_beams, num_answers)
        model = model
        generated_ids = model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            min_length=min_len,
            max_length=max_len,
            do_sample=do_sample,
            early_stopping=True,
            num_beams=1 if do_sample else n_beams,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            num_return_sequences=num_answers,
            decoder_start_token_id=tokenizer.bos_token_id,
        )
        return [tokenizer.decode(ans_ids, skip_special_tokens=True).strip() for ans_ids in generated_ids]

    def predict(self, dataset):
        predicted = []
        reference = []
        st_time = time()
        # Generate answers for the full test set
        for step in range(len(dataset)):
            # create support document with the dense index
            question = dataset[step]['question']
            support_doc = "<P> " + " <P> ".join([str(p) for p in dataset[step]["ctxs"]])
            # concatenate question and support document into BART input
            question_doc = "question: {} context: {}".format(question, support_doc)
            # generate an answer with beam search
            answer = self.generate(question_doc, self.model, self.tokenizer,
                        num_answers=1,num_beams=8,min_len=self.args.min_ans_length,
                        max_len=self.args.max_ans_length,max_input_length=self.args.max_input_length,device=self.args.device)
            predicted += [answer[0]]
            reference += [dataset[step]['answers'][0]]
            if step % self.args.eval_print_freq == 0 or step == (len(dataset) - 1):
                if self.args.logger:
                    self.logger.info("{:5d} of {:5d} \t -- {:.3f}".format(step, len(dataset) - 1, time() - st_time))
                elif self.args.is_notebook:
                    print("{:5d} of {:5d} \t -- {:.3f}".format(step, len(dataset) - 1, time() - st_time))

        return predicted, reference

    def evaluate(self, predicted, reference):
        stemmer = PorterStemmer()
        rouge = Rouge()
        nlpp = English()
        tokenizer = nlpp.tokenizer

        def compute_rouge_eli5(compare_list):
            preds = [" ".join([stemmer.stem(str(w))for w in tokenizer(pred)])for gold, pred in compare_list]
            golds = [" ".join([stemmer.stem(str(w))for w in tokenizer(gold)])for gold, pred in compare_list]
            scores = rouge.get_scores(hyps=preds, refs=golds, avg=True)
            return scores


        compare_list = [(g, p) for p, g in zip(predicted, reference)]
        scores = compute_rouge_eli5(compare_list)
        df = pd.DataFrame({
            'rouge1': [scores['rouge-1']['p'], scores['rouge-1']['r'], scores['rouge-1']['f']],
            'rouge2': [scores['rouge-2']['p'], scores['rouge-2']['r'], scores['rouge-2']['f']],
            'rougeL': [scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f']],
        }, index=[ 'P', 'R', 'F'])

        df.style.format({'rouge1': "{:.4f}", 'rouge2': "{:.4f}", 'rougeL': "{:.4f}"})

        if self.args.logger:
            self.logger.info(tabulate(df, headers = 'keys', tablefmt = 'psql'))
        elif self.args.is_notebook:
            print(tabulate(df, headers = 'keys', tablefmt = 'psql'))
