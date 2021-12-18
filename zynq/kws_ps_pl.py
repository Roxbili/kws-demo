#-*- encoding: utf-8 -*-

import argparse
import socket
import os, sys
import numpy as np
import functools
import mmap
from multiprocessing import Process

# from python_speech_features import mfcc


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

class BRAM(object):
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

        # print("Data: \n%s" % data)
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
        data = []
        for i in range(len):
            data.append(map_.read_byte())
        data = np.array(data, dtype=np.uint8)
        return data

    def read(self, len, start=None, map_id=0) -> np.ndarray:
        if map_id == 0:
            map_ = self.map0
        elif map_id == 1:
            map_ = self.map1

        if start != None:
            map_.seek(start)

        # read，和pl侧联调的时候用这段代码
        data = map_.read(len)
        data = np.frombuffer(data, dtype=np.uint8).copy()
        return data

class PSPLTalk(object):
    def __init__(self):
        self.model_dir = 'test_log/mobilenetv3_quant_mfcc_gen'
        self.words_list = "silence,unknown,yes,no,up,down,left,right,on,off,stop,go".split(',')
        self.bram = BRAM()
    
    def reset_flag(self):
        self.bram.write(b'\x00\x00\x00\x00', start=0)   # init, input
        self.bram.write(b'\x00\x00\x00\x00', start=0, map_id=1) # result
        self.bram.write(b'\x00\x00\x00\x00', start=4, map_id=1) # audio
        print('reset flag...')

class InputDataToBram(PSPLTalk):
    def __init__(self, mode):
        super(InputDataToBram, self).__init__()
        if mode == 'sdData':
            self.data_path = self.get_sdData()
            self.sendInputData = self.send_sdData

    def get_sdData(self):
        data_dir = os.path.join(self.model_dir, 'input_data')
        data_path = os.listdir(data_dir)
        data_path.sort()
        full_path = []
        for filename in data_path:
            path = os.path.join(data_dir, filename)
            full_path.append(path)
        print(full_path[:10])
        return full_path

    def bin2bram(self, bram, path, start=None, map_id=0):
        print('Data path: %s' % path)
        data = np.fromfile(path, dtype=np.uint8)
        bram.write(data, start=start, map_id=map_id)
        
        # print result
        # result = bram.read_oneByOne(data.nbytes, start=start)
        # print('Read:')
        # print(result)
        # print('')
    
    def info2bram(self):
        bin2bram_ = functools.partial(self.bin2bram, bram=self.bram)

        ################## flags ##################
        # init saved
        # self.bram.write(b'\x01\x01\x00\x00', start=0)
        # print('init saved')
        
        ################## network info ##################
        bin2bram_(path=os.path.join(self.model_dir, 'bin/structure_info.bin'), start=0x0004)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/weight_info.bin'), start=0x0010)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/feature_info.bin'), start=0x003C)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/scale_int.bin'), start=0x006C)

        ################## network weight ##################
        bin2bram_(path=os.path.join(self.model_dir, 'bin/stem_conv_weights.bin'), start=0x0090)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_1_expansion_weights.bin'), start=0x0390)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_1_depthwise_weights.bin'), start=0x0590)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_1_projection_weights.bin'), start=0x0790)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_2_expansion_weights.bin'), start=0x0990)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_2_depthwise_weights.bin'), start=0x0B90) 
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_2_projection_weights.bin'), start=0x0D90)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_3_expansion_weights.bin'), start=0x1190)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_3_depthwise_weights.bin'), start=0x1990)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_3_projection_weights.bin'), start=0x2190)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/Conv2D_weights.bin'), start=0x2990)

        ################## network bias ##################
        bin2bram_(path=os.path.join(self.model_dir, 'bin/stem_conv_bias.bin'), start=0x2B10)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_1_expansion_bias.bin'), start=0x2B50)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_1_depthwise_bias.bin'), start=0x2BD0)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_1_projection_bias.bin'), start=0x2C50)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_2_expansion_bias.bin'), start=0x2C90)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_2_depthwise_bias.bin'), start=0x2D10)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_2_projection_bias.bin'), start=0x2D90)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_3_expansion_bias.bin'), start=0x2E10)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_3_depthwise_bias.bin'), start=0x2F10)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/inverted_residual_3_projection_bias.bin'), start=0x3010)
        bin2bram_(path=os.path.join(self.model_dir, 'bin/Conv2D_bias.bin'), start=0x3090)

        ################## set network info flag 1 ##################
        self.bram.write(b'\x01', start=0)

    def sendData(self, data):
        '''send input data to bram'''
        # monitor input flag
        input_flag = self.bram.read_oneByOne(1, start=0x0)
        if input_flag[0] == 1:
            # save input data to bram
            self.bram.write(data, start=0x30C0)
            # set input flag
            self.bram.write(b'\x03', start=0x0)
            return True
        return False

    def send_sdData(self):
        while True:
            for input_data_path in self.data_path:
                input_data = np.load(input_data_path)
                self.sendData(input_data)

class Result(PSPLTalk):
    def __init__(self):
        super(Result, self).__init__()
        self.soct = self._init_soct()

    def _init_soct(self):
        # address = ('127.0.0.1', 6887)  # 服务端地址和端口
        # address = ('192.168.2.151', 6887)  # 服务端地址和端口
        address = ('192.168.2.116', 6887)  # 服务端地址和端口
        # address = ('10.130.147.227', 6887)  # 服务端地址和端口
        soct = Soct(address)
        return soct

    def send_result(self):
        while True:
            result_flag = self.bram.read_oneByOne(1, start=0x0, map_id=1)
            if result_flag[0] == 1:
                # reset result flag
                self.bram.write(b'\x00\x00\x00\x00', start=0x0, map_id=1)

                # get result
                result = self.bram.read_oneByOne(12, start=0x8, map_id=1)
                # send result
                word = self.words_list[np.argmax(result)]
                self.soct.send(word)
                print('send word %s' % word)


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
    input_object = InputDataToBram(args.mode)
    result_object = Result()

    ##################### reset flag ##################### 
    input_object.reset_flag()

    ##################### send network info ##################### 
    input_object.info2bram()

    ##################### run 2 process ##################### 
    p1 = Process(target=input_object.sendInputData)
    p1.daemon = True    # if main process finish, kill the subprocess instead of waitting
    p2 = Process(target=result_object.send_result)
    p2.daemon = True

    print('Start listening...')
    p1.start()
    p2.start()
    p1.join()
    p2.join()