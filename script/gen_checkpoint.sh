#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python create_quant_checkpoint.py \
    --model_architecture mobilenet-v3 \
    --dct_coefficient_count 10 \
    --window_size_ms 40 \
    --window_stride_ms 20 \
    --learning_rate 0.001,0.0001 \
    --how_many_training_steps 5000,2500 \
    --summaries_dir test_log/mobilenetv3_quant_mfcc_gen \
    --train_dir test_log/mobilenetv3_quant_mfcc_gen \
    --eval_step_interval 1200 \
    --quant \
    --bits 8 \
    --start_checkpoint test_log/mbnetv3_quant_8bit_mfcc/best/mobilenet-v3.ckpt-5400 \
    --data_dir /home/LAB/leifd/lfd/nnx-kws-ne001/speech_dataset