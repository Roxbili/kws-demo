#!/usr/bin/env bash

python get_kernel_feature_area/test_sdcard_numpy.py \
    --save_layers_output \
    --save_layers_input_feature \

python get_kernel_feature_area/npy2txt.py \