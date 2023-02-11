from torch.utils.data import Dataset

import json


def load_data(data_path = None):
    with open(data_path) as openfile:
        data = json.load(openfile)
    return data

def format_docs(data):
    result = {}
    for example in data:
        context = "<P> " + " <P> ".join(example['ctxs'])
        result[example['question_id']] = context
    return result
  
class ELI5DatasetS2S(Dataset): # modified
    def __init__(
        self, examples_array, make_doc_fun=None, document_cache=None):
        self.data = examples_array
        self.make_doc_function = make_doc_fun
        self.document_cache = {} if document_cache is None else document_cache
        assert not (make_doc_fun is None and document_cache is None)
        # make index of specific question-answer pairs from multi-answers
        self.qa_id_list = [(i, 0) for i in range(len(self.data))]

    def __len__(self):
        return len(self.qa_id_list)

    def make_example(self, idx):
        i, j = self.qa_id_list[idx]
        example = self.data[i]
        question = example['question']
        answer = example['answers'][j]
        q_id = example['question_id']
        if self.make_doc_function is not None:
            self.document_cache[q_id] = self.document_cache.get(q_id, self.make_doc_function(example['question']))
        document = self.document_cache[q_id]
        in_st = "question: {} context: {}".format(
            question.lower().replace(" --t--", "").strip(), document.lower().strip(),
        )
        out_st = answer
        return (in_st, out_st)

    def __getitem__(self, idx):
        return self.make_example(idx)
