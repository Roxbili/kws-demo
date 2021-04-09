#!/usr/bin/env bash

# Baseline Accuracy: 93.7

CUDA_VISIBLE_DEVICES=1 python train_quant.py \
    --model_architecture mobilenet-v3 \
    --dct_coefficient_count 10 \
    --window_size_ms 40 \
    --window_stride_ms 20 \
    --learning_rate 0.01,0.001,0.0001 \
    --how_many_training_steps 1800,1800,1800 \
    --summaries_dir test_log/mbnetv3_quant_8bit_log \
    --train_dir test_log/mbnetv3_quant_8bit \
    --eval_step_interval 200 \
    --quant \
    --bits 8 \
    # --start_checkpoint ./mbnetv3_quant_8bit/best/mobilenet-v3.ckpt-5400 \

# CUDA_VISIBLE_DEVICES=0 python train_quant.py \
#     --model_architecture mobilenet-v3 \
#     --dct_coefficient_count 10 \
#     --window_size_ms 40 \
#     --window_stride_ms 20 \
#     --learning_rate 0.01,0.001,0.0001 \
#     --how_many_training_steps 1800,1800,1800 \
#     --summaries_dir mbnetv3_quant_4bit_log \
#     --train_dir mbnetv3_quant_4bit \
#     --eval_step_interval 1200 \
#     --quant \
#     --bits 4

