#!/bin/sh
python run.py \
  --output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 \
  --model_type=roberta \
  --tokenizer_name=/root/workspace/graphcodebert-base \
  --model_name_or_path=/root/workspace/graphcodebert-base \
	--do_eval --do_test --do_train \
  --train_data_file=../dataset/train.jsonl \
  --eval_data_file=../dataset/valid.jsonl \
  --test_data_file=../dataset/test.jsonl \
	--block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 2>&1 | tee $logp/training_log.txt
