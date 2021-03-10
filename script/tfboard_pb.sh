#!/usr/bin/env bash

python tfboard_pb.py \
    --model_path ./mobilenetv3_fp_freeze_tflite/frozen_graph.pb \
    --summary_dir mobilenetv3_fp_freeze_tflite_log