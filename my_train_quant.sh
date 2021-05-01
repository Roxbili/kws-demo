#!/usr/bin/env bash

# Baseline Accuracy: 93.7

python train_quant.py \
    --model_architecture mobilenet-v3 \
    --dct_coefficient_count 10 \
    --window_size_ms 40 \
    --window_stride_ms 20 \
    --learning_rate 0.01,0.001,0.0001 \
    --how_many_training_steps 1800,1800,1800 \
    --summaries_dir mbnetv3_quant_8bit_log \
    --train_dir mbnetv3_quant_8bit \
    --eval_step_interval 200 \
    --quant \
    --bits 8 \
