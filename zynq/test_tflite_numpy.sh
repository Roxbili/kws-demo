#!/usr/bin/env bash

python test_tflite_numpy.py \
    --model_architecture mobilenet-v3 \
    --dct_coefficient_count 10 \
    --batch_size 1 \
    --window_size_ms 40 \
    --window_stride_ms 20 \
    --model_size_info 4 16 10 4 2 2 16 3 3 1 1 2 32 3 3 1 1 2 32 5 5 1 1 2 \
    --testing_mode simulate \
    --tflite_path test_log/mobilenetv3_quant_mfcc_gen/symmetric_8bit_mean220_std0.97.lite \
