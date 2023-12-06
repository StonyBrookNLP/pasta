#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 nohup python ./code/modelling_t5.py \
# -task_list "6" \
# -num_epochs 2 \
# -eval_steps 1000 \
# -batch_size 12 \
# -model_type "t5-base" \
# -random_seed 0 \
# -lr 1e-4 \
# -wt_decay 1e-6 > ./logs/t5-base-task6.log &


# CUDA_VISIBLE_DEVICES=1 nohup python ./code/modelling_t5.py \
# -task_list "7" \
# -num_epochs 2 \
# -eval_steps 1000 \
# -batch_size 12 \
# -model_type "t5-base" \
# -random_seed 0 \
# -lr 1e-4 \
# -wt_decay 1e-6 > ./logs/t5-base-task7.log &

# CUDA_VISIBLE_DEVICES=2 nohup python ./code/modelling_t5.py \
# -task_list "8" \
# -num_epochs 2 \
# -eval_steps 1000 \
# -batch_size 16 \
# -model_type "t5-base" \
# -random_seed 0 \
# -lr 1e-4 \
# -wt_decay 1e-6 > ./logs/t5-base-task8.log &

# CUDA_VISIBLE_DEVICES=0 python ./code/modelling_BERT_RoBERTa.py \
# -task_list "8" \
# -num_epochs 2 \
# -eval_steps 1000 \
# -batch_size 16 \
# -model_type "bert-base-uncased" \
# -random_seed 1231 \
# -lr 5e-6 \
# -task8_setting "1" \
# -wt_decay 1e-6 > ./logs/bert-base-task8.log &

# CUDA_VISIBLE_DEVICES=0 nohup python ./code/inference_BERT.py \
# -task_list "8" \
# -model_type "bert-base-uncased" \
# -task8_setting "1" \
# -checkpoint ./model_checkpoint/t_8_1_m_bert-base-uncased_b_16_lr_5e-06_w_1e-06_s_1231_epoch_1.pt &

# CUDA_VISIBLE_DEVICES=1 nohup python ./code/inference_T5.py \
# -task_list "8" \
# -model_type "t5-base" \
# -task8_setting "1" \
# -checkpoint ./model_checkpoint/t_8_1_m_t5-base_b_16_lr_0.0001_w_1e-06_s_0_epoch_1.pt &

# CUDA_VISIBLE_DEVICES=0 nohup python ./code/inference_BERT.py \
# -task_list "8" \
# -model_type "bert-base-uncased" \
# -task8_setting "1" \
# -checkpoint ./model_checkpoint/t_8_1_m_bert-base-uncased_b_16_lr_5e-06_w_1e-06_s_1231_epoch_1.pt &

# CUDA_VISIBLE_DEVICES=1 nohup python ./code/inference_T5.py \
# -task_list "8" \
# -model_type "t5-base" \
# -task8_setting "1" \
# -checkpoint ./model_checkpoint/t_8_1_m_t5-base_b_16_lr_0.0001_w_1e-06_s_0_epoch_1.pt &

CUDA_VISIBLE_DEVICES=0 nohup python ./code/inference_T5.py \
-task_list "6" \
-model_type "t5-base" \
-checkpoint "./model_checkpoint/t_6_m_t5-base_b_12_lr_0.0001_w_1e-06_s_0_epoch_1.pt" \
-task8_setting "1" > logs/_t5b_t_6_op.log &

CUDA_VISIBLE_DEVICES=1 nohup python ./code/inference_T5.py \
-task_list "7" \
-model_type "t5-base" \
-checkpoint "./model_checkpoint/t_7_m_t5-base_b_12_lr_0.0001_w_1e-06_s_0_epoch_1.pt" \
-task8_setting "1" > logs/_t5b_t_7_op.log &

CUDA_VISIBLE_DEVICES=2 nohup python ./code/inference_T5.py \
-task_list "8" \
-model_type "t5-base" \
-checkpoint "./model_checkpoint/t_8_1_m_t5-base_b_16_lr_0.0001_w_1e-06_s_0_epoch_1.pt" \
-task8_setting "1" > logs/_t5b_t_8_1_op.log &