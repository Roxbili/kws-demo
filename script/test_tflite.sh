#!/usr/bin/env bash

python test_tflite.py \
    --model_architecture mobilenet-v3 \
    --dct_coefficient_count 10 \
    --batch_size 1 \
    --window_size_ms 40 \
    --window_stride_ms 20 \
    --model_size_info 4 16 10 4 2 2 16 3 3 1 1 2 32 3 3 1 1 2 32 5 5 1 1 2 \
    --testing_mode simulate \
    --tflite_path test_log/mobilenetv3_quant_gen/swiftnet-uint8.lite \
    # --tflite_path test_log/mobilenetv3_quant_eval/uint8input_8bit_calc_mean220_std0.89.lite \
    # --tflite_path test_log/mobilenetv3_quant_eval/layers_lite_model/Conv2D.lite \
    # --tflite_path test_log/mobilenetv3_quant_eval/layers_lite_model/AvgPool.lite \
    # --tflite_path test_log/mobilenetv3_quant_eval/layers_lite_model/inverted_residual_3_add.lite \
    # --tflite_path tflite_factory/mbnetv3-8b-92.37.lite \
