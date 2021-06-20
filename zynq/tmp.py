#-*- encoding: utf-8 -*-

import argparse
from tkinter.constants import TRUE
import numpy as np
import tkinter as tk 
from tkinter.ttk import Label
from multiprocessing import Process, Queue
import time

class App(object):
    def __init__(self, queue):
        self.q = queue

        self.root = tk.Tk()
        self.word = Label(self.root)
        self.txt_placeholder = tk.StringVar()
        # self._set_text('###')
        self._set_text('yesgo')
        
        color = '#1C1C1C'
        self._set_root(color)
        self._set_label(color)

        # self.get_text()

    def mainloop(self):
        self.root.mainloop()

    def _set_root(self, color):
        self.root.geometry('200x50')
        self.root.title('Keywords spotting')
        self.root.config(background=color)
    
    def _set_label(self, color):
        self.word.config(
            width=20,
            font=("Times", 40, 'bold'), 
            textvariable=self.txt_placeholder,
            background=color, 
            foreground='#FCFAF2'
        )
        # self.txt_placeholder.set('unknown')

        # lbl = Label(root, font = ('calibri', 40, 'bold'), 
        #             background = 'purple', 
        #             foreground = 'white') 
        self.word.pack(anchor='center', ipady=10)
    
    def _set_text(self, txt):
        self.txt_placeholder.set(txt)
    
    def get_text(self):
        if not self.q.empty():
            txt = self.q.get()
            self._set_text(txt)
        self.word.after(1, self.get_text)

def push(q):
    for i in range(100):
        q.put(str(i))
        print('push %d' % i)
        time.sleep(0.5)

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
    q = Queue()
    app = App(q)

    # p = Process(target=push, args=[q])
    # p.start()
    app.mainloop()
    # p.join()