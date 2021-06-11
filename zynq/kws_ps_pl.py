#-*- encoding: utf-8 -*-

import argparse
import socket
import os, sys
import numpy as np
import functools
import mmap

from python_speech_features import mfcc


class Soct(object):
    def __init__(self, address):
        # buffer_size = 16
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.s.connect(address)  # 尝试连接服务端
        except Exception as e:
            # print('[!] Server not found or not open')
            print(e)
            sys.exit(0)
    
    def __del__(self):
        self.s.close()

    def send(self, info):
        self.s.sendall(info.encode())

class BRAM:
    def __init__(self):
        self.BRAM_CTRL_0 = 0x40000000	# AXI_bram_ctrl_0的物理地址
        self.BRAM_CTRL_1 = 0x41000000	# AXI_bram_ctrl_1的物理地址
        self.DATA_LEN_0 = 16384 			# 写入和读取的数据长度，单位字节
        self.DATA_LEN_1 = 32768 			# 写入和读取的数据长度，单位字节

        self.map0, self.map1 = self.mapping('/dev/mem')

    def __del__(self):
        os.close(self.file)
        self.map0.close()
        self.map1.close()

    def mapping(self, path):
        self.file = os.open(path, os.O_RDWR | os.O_SYNC)
        map0 = mmap.mmap(
            self.file, 
            self.DATA_LEN_0,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
            offset=self.BRAM_CTRL_0
        )
        map1 = mmap.mmap(
            self.file, 
            self.DATA_LEN_1,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
            offset=self.BRAM_CTRL_1
        )
        return map0, map1
    
    def write(self, data, start=None, map_id=0):
        '''写入数据
            由于数据位宽32bit，因此最好以4的倍数Byte写入
        '''
        # print("Write data via AXI_bram_ctrl_0")
        if map_id == 0:
            map_ = self.map0
        elif map_id == 1:
            map_ = self.map1

        print("Data: \n%s" % data)
        if start != None:
            map_.seek(start)
        map_.write(data)
    
    def read_oneByOne(self, len, start=None, map_id=0):
        # print("Read data from BRAM via AXI_bram_ctrl_1")
        if map_id == 0:
            map_ = self.map0
        elif map_id == 1:
            map_ = self.map1

        if start != None:
            map_.seek(start)
        data = ""
        for i in range(len):
            data += map_.read_byte()
        data = np.frombuffer(bytes(data), dtype=np.uint8)
        return data


def read_bin(path):
    data = np.fromfile(path, dtype=np.uint8)
    return data

def bin2bram(bram, path, start=None, map_id=0):
    print('Path: %s' % path)
    data = read_bin(path)
    bram.write(data, start=start, map_id=map_id)
    
    # print result
    result = bram.read_oneByOne(data.nbytes, start=start)
    print('Read:')
    print(result)
    print('')

def info2bram(bram):
    bin2bram_ = functools.partial(bin2bram, bram=bram)

    ################## flags ##################
    # init saved
    bram.write(b'\x01\x01\x00\x00', start=0)
    print('init saved')
    result = bram.read_oneByOne(4, start=0, map_id=1)
    print(result)
    
    ################## network info ##################
    bin2bram_(path=os.path.join(model_dir, 'bin/structure_info.bin'), start=4)
    bin2bram_(path=os.path.join(model_dir, 'bin/weight_info.bin'), start=16)
    bin2bram_(path=os.path.join(model_dir, 'bin/feature_info.bin'), start=60)
    bin2bram_(path=os.path.join(model_dir, 'bin/scale_int.bin'), start=108)

    ################## network parameters ##################
    bin2bram_(path=os.path.join(model_dir, 'bin/stem_conv_weights.bin'), start=144)
    bin2bram_(path=os.path.join(model_dir, 'bin/stem_conv_bias.bin'), start=912)

    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_1_expansion_weights.bin'), start=976)
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_1_expansion_bias.bin'), start=1488)
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_1_depthwise_weights.bin'), start=1616)
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_1_depthwise_bias.bin'), start=2128)
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_1_projection_weights.bin'), start=2256)
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_1_projection_bias.bin'), start=2768)

    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_2_expansion_weights.bin'), start=2832)
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_2_expansion_bias.bin'), start=3344)
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_2_depthwise_weights.bin'), start=3472) 
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_2_depthwise_bias.bin'), start=3984)
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_2_projection_weights.bin'), start=4112)
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_2_projection_bias.bin'), start=5136)

    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_3_expansion_weights.bin'), start=5264)
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_3_expansion_bias.bin'), start=7312)
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_3_depthwise_weights.bin'), start=7568)
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_3_depthwise_bias.bin'), start=9616)
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_3_projection_weights.bin'), start=9872)
    bin2bram_(path=os.path.join(model_dir, 'bin/inverted_residual_3_projection_bias.bin'), start=11920)

    bin2bram_(path=os.path.join(model_dir, 'bin/Conv2D_weights.bin'), start=12048)
    bin2bram_(path=os.path.join(model_dir, 'bin/Conv2D_bias.bin'), start=12432)

    ################## set network info flag 1 ##################
    bram.write(b'\x01\x00\x00\x00', start=0)

def one_image(bram, soct):
    # send input data to bram
    input_data_path = os.path.join(model_dir, 'bin/go.bin')
    bin2bram(bram, input_data_path, start=12480)
    # set input data flag 1
    bram.write(b'\x01\x01\x00\x00', start=0)

    # wordlist
    words_list = "silence,unknown,yes,no,up,down,left,right,on,off,stop,go".split(',')

    # monitor result flag
    while True:
        result_flag = bram.read_oneByOne(1, start=0, map_id=1)
        if result_flag[0] == True:
            result = bram.read_oneByOne(12, start=4, map_id=1)
            word = words_list[np.argmax(result)]
            soct.send(word)
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        default='sdData',
    )
    args = parser.parse_args()

    ##################### init ##################### 
    model_dir = 'test_log/mobilenetv3_quant_mfcc_gen'

    # address = ('127.0.0.1', 6887)  # 服务端地址和端口
    # address = ('192.168.2.151', 6887)  # 服务端地址和端口
    address = ('192.168.2.117', 6887)  # 服务端地址和端口
    # address = ('10.130.147.227', 6887)  # 服务端地址和端口
    soct = Soct(address)

    bram = BRAM()

    ##################### send network info ##################### 
    info2bram(bram)

    ##################### run ##################### 
    if args.mode == 'sdData':
        imple_func = one_image
    else:
        print('No mode is found.')
        sys.exit(0)

    imple_func()
    print('Done!')