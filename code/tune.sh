#!/bin/sh
python tune.py \
  --output_dir=./tuning_results \
  --model_type=roberta \
  --tokenizer_name=/root/workspace/graphcodebert-base \
  --model_name_or_path=/root/workspace/graphcodebert-base \
  --train_data_file=../dataset/train.jsonl \
  --eval_data_file=../dataset/valid.jsonl \
  --test_data_file=../dataset/test.jsonl \
  --n_trials 30 --subset 0.3 --epoch 10 \
  --block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 \
  --seed 123456 2>&1 | tee tuning_log.txt
