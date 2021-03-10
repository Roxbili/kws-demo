#!/usr/bin/env bash

# Baseline Accuracy: 93.7
# Final Accuracy: 92.54
# MFCC特征，window_size_ms为窗口大小，window_stride_ms为步长

python train_quant.py \
    --model_architecture mobilenet-v3 \
    --dct_coefficient_count 10 \
    --window_size_ms 40 \
    --window_stride_ms 20 \
    --learning_rate 0.001,0.0001,0.00001 \
    --how_many_training_steps 15000,5000,2500 \
    --summaries_dir mbnetv3_fp_log \
    --train_dir mbnetv3_fp \
    --eval_step_interval 1200