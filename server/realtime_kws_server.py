#-*- encoding: utf-8 -*-

from flask import Flask, render_template
from flask_socketio import SocketIO
import socket
import threading
import time

############# socket io parameter #############

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

############# socket with zynq parameter #############

address = ('127.0.0.1', 6887)  # 服务端地址和端口
buffer_size = 16
word = ''

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(address)  # 绑定服务端地址和端口
s.listen(2)
conn, addr = s.accept()  # 返回客户端地址和一个新的 socket 连接
print('[+] Connected with', addr)


############# socket with zynq run #############

def kws_result():
    global word

    while True:
        word = conn.recv(buffer_size)  # 单位字节
        word = word.decode()
        if word == '###':
            conn.close()
            s.close()
            break
        time.sleep(1)
        # print('[Received]', word)
        # send = input('Input: ')
        # conn.sendall(send.encode())

# create thread to receive kws result from zynq
kws_thread = threading.Thread(target=kws_result)
kws_thread.start()

############# socket io run #############

@app.route('/')
def hello_world():
    return render_template('index.html')

@socketio.on('connect', namespace='/message')
def send_message():
    socketio.start_background_task(target=background_func)

def background_func():
    while True:
        socketio.emit('server_response', {'data': word}, namespace='/message')
        socketio.sleep(2)



if __name__ == '__main__':
    socketio.run(app)