#-*- encoding: utf-8 -*-

import numpy as np
import os, sys

# 设置信息全打印
np.set_printoptions(threshold=np.inf)

log_dir = 'get_kernel_feature_area/log'
layers_output_dir = os.path.join(log_dir, 'output')
layers_input_dir = os.path.join(log_dir, 'input')

class Saver(object):
    def __init__(self):
        pass

    def print_data_to_file(self, path):
        print('=' * 50, file=self.f)
        print('=' * 50, file=self.f)
        print(path, file=self.f)
        
        data = np.load(path)
        print('Data shape: {}\n'.format(data.shape), file=self.f)

        print(data, '\n', file=self.f)

    def save(self, src_dir, dst_txt):
        '''
            Args:
                mode: output | input
        '''
        self.f = open(dst_txt, 'w')
        npy_list = os.listdir(src_dir)
        npy_list.sort()
        for item in npy_list:
            if 'inverted_residual' in item:
                self.print_data_to_file(os.path.join(src_dir, item))

        self.f.close()


saver = Saver()
saver.save(layers_output_dir, os.path.join(log_dir, 'output.txt'))
saver.save(layers_input_dir, os.path.join(log_dir, 'input.txt'))

print('See get_kernel_feature_area/log/input.txt and output.txt')