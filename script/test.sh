#!/usr/bin/env bash

python test.py \
    --model_architecture mobilenet-v3 \
    --dct_coefficient_count 10 \
    --window_size_ms 40 \
    --window_stride_ms 20 \
    --model_size_info 4 16 10 4 2 2 16 3 3 1 1 2 32 3 3 1 1 2 32 5 5 1 1 2 \
    --checkpoint ./mbnetv3_quant_8bit/best/mobilenet-v3.ckpt-5400