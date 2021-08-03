#-*- encoding: utf-8 -*-

import time
import argparse
import numpy as np
import tkinter as tk 
from tkinter.ttk import Label
from kws_ps_pl import BRAM, PSPLTalk, InputDataToBram
from multiprocessing import Process

class timeRecorder(object):
    def __init__(self):
      self.total_time = 0.
      self.counter = 0
    
    def start(self):
      self.start_time = time.time()

    def end(self):
      self.total_time += time.time() - self.start_time
      self.counter += 1

    def get_total_time(self):
      return self.total_time

    def get_avg_time(self):
      return self.total_time / self.counter

class App(PSPLTalk):
    def __init__(self, args):
        super(App, self).__init__()
        self.input_object = InputDataToBram(args.mode)
        self.input_object.reset_flag()   # 初始化标记位
        self.input_object.info2bram()    # 发送基本的参数
        self.input_data_path = iter(self.input_object.data_path)    # 创建输入数据路径的迭代器

        self.timer = timeRecorder()

        self.root = tk.Tk()
        self.word = Label(self.root)
        self.txt_placeholder = tk.StringVar()
        self._set_text('###')
        
        color = '#1C1C1C'
        self._set_root(color)
        self._set_label(color)
        self.first_scan = True  # 第一轮mainloop先把组件显示出来
    
    def mainloop(self):
        self.root.mainloop()
    
    def _set_root(self, color):
        self.root.geometry('200x60')
        self.root.title('Keywords spotting')
        self.root.config(background=color)
    
    def _set_label(self, color):
        self.word.config(
            width = 7,
            font=("Times", 40, 'bold'), 
            textvariable=self.txt_placeholder,
            background=color, 
            foreground='#FCFAF2'
        )
        # self.txt_placeholder.set('unknown')

        # lbl = Label(root, font = ('calibri', 40, 'bold'), 
        #             background = 'purple', 
        #             foreground = 'white') 
        self.word.pack(anchor='center', ipady=5)
    
    def _set_text(self, txt):
        self.txt_placeholder.set(txt)
    
    def show_result(self):
        # 第一轮mainloop先显示组件
        if self.first_scan:
            self.word.after(1000, self.show_result)
            self.first_scan = False
            return

        # 首先拿到数据
        path = next(self.input_data_path)     # 遍历数据集
        # path = self.input_object.data_path[0]   # 测试用，仅看0_no.npy
        input_data = np.load(path)
        # 接着监测标记位是否改变，是的话发送数据，否则阻塞
        while not self.input_object.sendData(input_data): pass

        while True:
            result_flag = self.bram.read_oneByOne(1, start=0x0, map_id=1)
            if result_flag[0] == 1:
                self.timer.start()

                # reset result flag
                self.bram.write(b'\x00\x00\x00\x00', start=0x0, map_id=1)
                # get result
                result = self.bram.read_oneByOne(12, start=0x4, map_id=1)
                # show result
                word = self.words_list[np.argmax(result)]
                self._set_text(word)
                print('path: %s, show word %s' % (path, word))

                self.timer.end()
                print('Total time: {}'.format(self.timer.get_total_time()))
                print('Average time: {}'.format(self.timer.get_avg_time()))
                self.word.after(1, self.show_result)    # 表示接着运行
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        default='sdData',
    )
    args = parser.parse_args()

    ##################### init ##################### 
    app = App(args)

    ##################### run 2 process ##################### 
    print('Start listening...')
    app.show_result()
    app.mainloop()