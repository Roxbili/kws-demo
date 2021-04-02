#!/usr/bin/env bash

python tfboard_pb.py \
    --model_path log/mobilenetv3_quant_eval/frozen_graph.pb \
    --summary_dir log/mobilenetv3_quant_eval