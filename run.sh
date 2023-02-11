#!/bin/sh
python3 train.py \
	--train_data /kaggle/input/eli5-10-doc/ELI5_train_10_doc.json \
	--eval_data /kaggle/input/eli5-10-doc/ELI5_val_10_doc.json \
	--seed 0 \
	--is_notebook True \
    --name experiment-1 \
    --logger False \
    --checkpoint_dir checkpoint \
    --model_name t5 \
    --model_size base \
	--num_epochs 1\
	--batch_size 8 \
	--max_input_length 1024 \
	--min_ans_length 64 \
	--max_ans_length 256 \
	--train_print_freq 1000 \
	--eval_print_freq 100 \
	--save_freq 10000 \
	--lr 2e-4 \
	--backward_freq 16 
