#-*- encoding: utf-8 -*-

import numpy as np
import queue
from multiprocessing import Process

from kws_ps_pl import BRAM
from test_tflite_numpy import simulate_net

class PL(object):
    def __init__(self):
        self.bram = BRAM()
        self.q = queue.Queue()

    def get_data_to_queue(self):
        for _ in range(100):    # 先保存100张图片到队列里面，毕竟这只是个模拟，实际未必这么操作
            self.bram.write(b'\x01\x00\x00\x00', start=0x4, map_id=1)
            # input flag == 1?
            while True:
                input_flag = self.bram.read_oneByOne(1, start=0x1)
                if input_flag == 1:
                    self.bram.write(b'\x00\x00\x00\x00', start=0x1)
                    break
            # read data
            data = self.bram.read_oneByOne(490, start=0x30C0)
            self.q.put(data)
        
    def inference(self):
        while not self.q.empty():
            data = self.q.get()
            output_uint8 = simulate_net(data)
            self.bram.write(output_uint8, start=0x8)
            self.bram.write(b'\x01\x00\x00\x00', start=0x0, map_id=1)

if __name__ == '__main__':
    pl = PL()
    p1 = Process(target=pl.get_data_to_queue)
    p2 = Process(target=pl.inference)

    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print('Done')