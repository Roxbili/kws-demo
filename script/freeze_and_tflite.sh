#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python freeze_and_tflite.py \
    --model_architecture mobilenet-v3 \
    --dct_coefficient_count 10 \
    --window_size_ms 40 \
    --window_stride_ms 20 \
    --model_dir log/fp_ver_all8bit \
    --model_size_info 4 16 10 4 2 2 16 3 3 1 1 2 32 3 3 1 1 2 32 5 5 1 1 2 \
    --checkpoint ./mbnetv3_fp/best/mobilenet-v3.ckpt-22500 \
    --quantize_type all