#-*- encoding: utf-8 -*-

import numpy as np
import os, sys
from functools import partial

layers_output_dir = 'get_kernel_feature_area/log/output'

# 设置信息全打印
np.set_printoptions(threshold=np.inf)

def print_data(f, path):
    # print('=' * 50)
    # print('=' * 50)
    # print(path)

    print('=' * 50, file=f)
    print('=' * 50, file=f)
    print(path, file=f)
    
    data = np.load(path)
    # print('Data shape: {}\n'.format(data.shape))
    print('Data shape: {}\n'.format(data.shape), file=f)

    # print(data, '\n')
    print(data, '\n', file=f)


with open('log.txt', 'w') as f:
    print_data_ = partial(print_data, f)

    # print_data_(os.path.join(layers_output_dir, 'input_data.npy'))
    print_data_(os.path.join(layers_output_dir, 'stem_conv.npy'))
    print_data_(os.path.join(layers_output_dir, 'inverted_residual_1_expansion.npy'))
    print_data_(os.path.join(layers_output_dir, 'inverted_residual_1_depthwise.npy'))
    print_data_(os.path.join(layers_output_dir, 'inverted_residual_1_projection.npy'))
    print_data_(os.path.join(layers_output_dir, 'inverted_residual_1_add.npy'))
    print_data_(os.path.join(layers_output_dir, 'inverted_residual_2_expansion.npy'))
    print_data_(os.path.join(layers_output_dir, 'inverted_residual_2_depthwise.npy'))
    print_data_(os.path.join(layers_output_dir, 'inverted_residual_2_projection.npy'))
    print_data_(os.path.join(layers_output_dir, 'inverted_residual_3_expansion.npy'))
    print_data_(os.path.join(layers_output_dir, 'inverted_residual_3_depthwise.npy'))
    print_data_(os.path.join(layers_output_dir, 'inverted_residual_3_projection.npy'))
    print_data_(os.path.join(layers_output_dir, 'inverted_residual_3_add.npy'))
    print_data_(os.path.join(layers_output_dir, 'AvgPool.npy'))
    print_data_(os.path.join(layers_output_dir, 'Conv2D.npy'))

    print('See log.txt')