#!/usr/bin/env bash

python create_quant_checkpoint.py \
    --model_architecture mobilenet-v3 \
    --dct_coefficient_count 10 \
    --window_size_ms 40 \
    --window_stride_ms 20 \
    --learning_rate 0.001,0.0001 \
    --how_many_training_steps 5000,2500 \
    --summaries_dir mobilenetv3_quant_log_eval \
    --train_dir mobilenetv3_quant_eval \
    --eval_step_interval 1200 \
    --quant \
    --bits 8 \
    --start_checkpoint ./mbnetv3_quant_8bit/best/