#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python test_pb.py \
    --model_architecture mobilenet-v3 \
    --dct_coefficient_count 10 \
    --window_size_ms 40 \
    --window_stride_ms 20 \
    --model_size_info 4 16 10 4 2 2 16 3 3 1 1 2 32 3 3 1 1 2 32 5 5 1 1 2 \
    --graph ./mobilenetv3_quant_eval/frozen_graph.pb \
    --training \