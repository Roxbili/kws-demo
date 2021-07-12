#-*- encoding: utf-8 -*-

import numpy as np
import os, sys

# 设置信息全打印
np.set_printoptions(threshold=np.inf)

log_dir = 'get_kernel_feature_area/log'
layers_output_dir = os.path.join(log_dir, 'output')
layers_input_dir = os.path.join(log_dir, 'input')
layers_output_before_bias_dir = os.path.join(log_dir, 'output_before_bias')

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
        '''save all npy data to one txt file'''
        self.f = open(dst_txt, 'w')
        npy_list = os.listdir(src_dir)
        npy_list.sort()
        for item in npy_list:
            if 'inverted_residual' in item:
                self.print_data_to_file(os.path.join(src_dir, item))

        self.f.close()

    def nptxt_data_to_file(self, npy_path, dst_dir, mode):
        basename = os.path.basename(npy_path).split('.')[0]
        data = np.load(npy_path)
        data = data.squeeze()
        if len(data.shape) == 3:
            if mode == 'channel_by_channel':
                # save per channels to one .txt file
                for channel_id in range(data.shape[-1]):
                    channel_file_name = basename + '_{}.txt'.format(channel_id)
                    channel_file_path = os.path.join(dst_dir, channel_file_name)
                    channel_data = data[..., channel_id]
                    np.savetxt(channel_file_path, channel_data, fmt='%d')

            elif mode == 'channel_scan_row':
                # save data sequence: channel -> scan row, one layer one txt file
                data = data.reshape(data.shape[0] * data.shape[1], -1)
                np.savetxt(os.path.join(dst_dir, basename + '.txt'), data, fmt='%d')

        else:
            filename = basename + '.txt'
            filepath = os.path.join(dst_dir, filename)
            np.savetxt(filepath, data)

    def save_output(self, src_dir, dst_dir, mode):
        '''one .npy file may produce multiple .txt file for multiple channels'''
        npy_list = os.listdir(src_dir)
        npy_list.sort()
        for item in npy_list:
            self.nptxt_data_to_file(os.path.join(src_dir, item), dst_dir, mode)


saver = Saver()

# save to one file
# saver.save(layers_output_dir, os.path.join(log_dir, 'output.txt'))
# saver.save(layers_input_dir, os.path.join(log_dir, 'input.txt'))
# print('See get_kernel_feature_area/log/input.txt and output.txt')

# save to mutiple files, one channel one file
# saver.save_output(os.path.join(layers_output_before_bias_dir, 'npy'), os.path.join(layers_output_before_bias_dir, 'txt'), mode='channel_by_channel')

# save to mutiple files, channle -> scan row sequence
# before add bias
saver.save_output(os.path.join(layers_output_before_bias_dir, 'npy'), os.path.join(layers_output_before_bias_dir, 'txt'), mode='channel_scan_row')
# real output
saver.save_output(os.path.join(layers_output_dir, 'npy'), os.path.join(layers_output_dir, 'txt'), mode='channel_scan_row')

print('Save txt files successfully')