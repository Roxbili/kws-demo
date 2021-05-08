#-*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import numpy as np
import time
import socket
import random
import wave
from pyaudio import PyAudio, paInt16 
from python_speech_features import mfcc

from layers import conv2d, depthwise_conv2d, relu, pooling

# os.chdir('../')

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

# address = ('127.0.0.1', 6887)  # 服务端地址和端口
# address = ('192.168.2.151', 6887)  # 服务端地址和端口
address = ('192.168.2.117', 6887)  # 服务端地址和端口
# address = ('10.130.147.227', 6887)  # 服务端地址和端口
soct = Soct(address)


def fp32_to_uint8(r):
    # method 1
    # s = (r.max() - r.min()) / 255.
    # z = 255. - r.max() / s
    # q = r / s + z

    # tf_mfcc
    # std_dev = 0.8934739293234528
    # mean_ = 220.81257374779565
    # q = r / std_dev + mean_
    # q = q.astype(np.uint8)

    # new_mfcc
    std_dev = 0.9671023485944863
    mean_ = 220.46072856666711
    q = r / std_dev + mean_
    q = q.astype(np.uint8)
    return q

def psf_mfcc(wav_data):
    # frequency_sampling, x = wavfile.read(wav_path)
    x = np.frombuffer(np.array(wav_data).tobytes(), np.int16)
    # print(x.shape, x.dtype)
    
    x = np.pad(x, (0, 16000 - x.shape[0]), mode='constant')

    if x.dtype == 'int16':
        nb_bits = 16 # -> 16-bit wav files
    elif x.dtype == 'int32':
        nb_bits = 32 # -> 32-bit wav files
    max_nb_bit = float(2 ** (nb_bits - 1))
    audio_signal = x / max_nb_bit # samples is a numpy array of float representing the samples
    clamp_audio = audio_signal.clip(-1., 1.)

    features_mfcc = mfcc(clamp_audio, samplerate=16000, 
                        winlen=0.04, winstep=0.02, numcep=10, nfilt=40, lowfreq=20, highfreq=4000,
                        nfft=800, appendEnergy=False, preemph=0, ceplifter=0)
    # print(features_mfcc.shape)

    # outputs = features_mfcc.flatten()
    # print(outputs.shape)
    # print(outputs)

    return features_mfcc

class Net(object):
    def __init__(self):

        self.s_iwr, self.s_add = self._init_parameter()
        self.weight, self.bias = self._init_weight()
    
    def _init_parameter(self):
        # tf mfcc parameters
        # bias_scale = np.array([0.0008852639002725482, 0.0035931775346398354, 0.00785899069160223, 0.0014689048985019326, 0.0015524440677836537, 0.0028435662388801575, 0.001141879241913557, 0.0007087105768732727, 0.009289528243243694, 0.0015117411967366934, 0.004092711955308914])
        # result_sacale = np.array([0.20100615918636322, 0.42823609709739685, 0.23841151595115662, 0.1732778549194336, 0.21222199499607086, 0.15781369805335999, 0.12740808725357056, 0.1111915186047554, 0.11338130384683609, 0.19232141971588135, 0.17540767788887024])
        # add_scale = np.array([0.1732778549194336, 0.20100615918636322, 0.26455792784690857, 0.19232141971588135, 0.12740808725357056, 0.20970593392848969])

        # new mfcc parameters V100
        # bias_scale = np.array([0.0005171183147467673, 0.0021205246448516846, 0.004102946724742651, 0.0007573990151286125, 0.0009573157876729965, 0.0045410459861159325, 0.0007452332065440714, 0.0003749248862732202, 0.0028607698623090982, 0.0014322539791464806, 0.0036672416608780622])
        # result_sacale = np.array([0.12663139402866364, 0.20024137198925018, 0.13141511380672455, 0.11106141656637192, 0.1328522115945816, 0.08316611498594284, 0.08792730420827866, 0.08202825486660004, 0.1061563566327095, 0.17049182951450348, 0.18540261685848236])
        # add_scale = np.array([0.11106141656637192, 0.12663139402866364, 0.13807182013988495, 0.17049182951450348, 0.08792730420827866, 0.20207594335079193])

        # new mfcc parameters ACT
        bias_scale = np.array([0.0006772454944439232, 0.0019126507686451077, 0.004039060324430466, 0.0009780717082321644, 0.0011637755669653416, 0.002527922624722123, 0.000784197065513581, 0.00036984056350775063, 0.0027576638385653496, 0.0018317087087780237, 0.003179859137162566])
        result_sacale = np.array([0.15135173499584198, 0.20287899672985077, 0.1442921757698059, 0.11213209480047226, 0.1550600677728653, 0.0902664065361023, 0.07894150912761688, 0.0978255569934845, 0.08960756659507751, 0.1850544661283493, 0.19603444635868073])
        add_scale = np.array([0.11213209480047226, 0.15135173499584198, 0.16829396784305573, 0.1850544661283493, 0.07894150912761688, 0.1915309578180313])

        scale = bias_scale / result_sacale
        scale = (np.round(scale * 2**10) / 2**10).astype(np.float32)
        add_scale = (np.round(add_scale * 2**10) / 2**10).astype(np.float32)

        s_iwr = {
            'stem_conv': scale[0], 
            'inverted_residual_1_expansion': scale[1], 'inverted_residual_1_depthwise': scale[2], 'inverted_residual_1_projection': scale[3], 
            'inverted_residual_2_expansion': scale[4], 'inverted_residual_2_depthwise': scale[5], 'inverted_residual_2_projection': scale[6], 
            'inverted_residual_3_expansion': scale[7], 'inverted_residual_3_depthwise': scale[8], 'inverted_residual_3_projection': scale[9], 
            'Conv2D': scale[10]
        }
        s_add = {
            'inverted_residual_1_add': add_scale[:3], 
            'inverted_residual_3_add': add_scale[3:], 
        }

        return (s_iwr, s_add)

    def _init_weight(self):
        # model_dir = 'test_log/mobilenetv3_quant_gen'
        model_dir = 'test_log/mobilenetv3_quant_mfcc_gen'
        weight = {}
        bias = {}

        weight['stem_conv'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_stem_conv_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
        bias['stem_conv'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_stem_conv_conv_Conv2D_Fold_bias.npy'))

        weight['inverted_residual_1_expansion'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
        bias['inverted_residual_1_expansion'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_expansion_conv_Conv2D_Fold_bias.npy'))

        weight['inverted_residual_1_depthwise'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'))
        bias['inverted_residual_1_depthwise'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_depthwise_depthwise_conv_Fold_bias.npy'))

        weight['inverted_residual_1_projection'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
        bias['inverted_residual_1_projection'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_projection_conv_Conv2D_Fold_bias.npy'))

        weight['inverted_residual_2_expansion'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
        bias['inverted_residual_2_expansion'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_expansion_conv_Conv2D_Fold_bias.npy'))

        weight['inverted_residual_2_depthwise'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'))
        bias['inverted_residual_2_depthwise'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_depthwise_depthwise_conv_Fold_bias.npy'))

        weight['inverted_residual_2_projection'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
        bias['inverted_residual_2_projection'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_projection_conv_Conv2D_Fold_bias.npy'))

        weight['inverted_residual_3_expansion'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
        bias['inverted_residual_3_expansion'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_expansion_conv_Conv2D_Fold_bias.npy'))

        weight['inverted_residual_3_depthwise'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'))
        bias['inverted_residual_3_depthwise'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_depthwise_depthwise_conv_Fold_bias.npy'))

        weight['inverted_residual_3_projection'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
        bias['inverted_residual_3_projection'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_projection_conv_Conv2D_Fold_bias.npy'))

        weight['Conv2D'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_fc_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
        bias['Conv2D'] = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_fc_conv_Conv2D_bias.npy'))

        return (weight, bias)

    def __call__(self, input_data):

        ################## stem conv ##################
        # print('stem conv')
        new_data = input_data.astype(np.float32)
        new_data = new_data - 221.

        weight = self.weight['stem_conv'].astype(np.float32)
        weight = weight - 128.
        weight = weight.transpose(1,2,0,3)

        bias = self.bias['stem_conv'].astype(np.float32)
        # print(weight)
        # print(bias)

        output = depthwise_conv2d(new_data, weight, stride=(2,2), pad="SAME")

        output += bias
        output = output * self.s_iwr['stem_conv']
        # output += 0.0035
        
        output = relu(output)
        output += 128
        output_uint8 = output.round()
        output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
        add_2 = output_uint8.copy()   # 给之后的做加法
        # print()

        ################## inverted residual 1 expansion ##################
        # print('inverted residual 1 expansion')
        new_data = output_uint8.astype(np.float32)
        new_data -= 128

        weight = self.weight['inverted_residual_1_expansion'].astype(np.float32)
        weight = weight - 128.
        weight = weight.transpose(1,2,3,0)
        # print(weight)

        bias = self.bias['inverted_residual_1_expansion'].astype(np.float32)
        # print(bias)

        output = conv2d(new_data, weight, stride=(1,1), pad="SAME")
        output = output + bias
        # output += 0.0074
        output = output * self.s_iwr['inverted_residual_1_expansion']
        
        output = relu(output)
        output += 128
        output_uint8 = output.round()
        output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
        # print()

        ################## inverted residual 1 depthwise ##################
        # print('inverted residual 1 depthwise')
        new_data = output_uint8.astype(np.float32)
        new_data -= 128

        weight = self.weight['inverted_residual_1_depthwise'].astype(np.float32)
        weight = weight - 128.
        weight = weight.transpose(1,2,3,0)
        # print(weight)

        bias = self.bias['inverted_residual_1_depthwise'].astype(np.float32)
        # print(bias)

        output = depthwise_conv2d(new_data, weight, stride=(1,1), pad="SAME")
        output = output + bias
        # output += 0.0301
        output = output * self.s_iwr['inverted_residual_1_depthwise']
        
        output = relu(output)
        output += 128
        output_uint8 = output.round()
        output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
        # print()

        ################## inverted residual 1 projection ##################
        # print('inverted residual 1 projection')
        new_data = output_uint8.astype(np.float32)
        new_data -= 128

        weight = self.weight['inverted_residual_1_projection'].astype(np.float32)
        weight = weight - 128.
        weight = weight.transpose(1,2,3,0)
        # print(weight)

        bias = self.bias['inverted_residual_1_projection'].astype(np.float32)
        # print(bias)

        output = conv2d(new_data, weight, stride=(1,1), pad="SAME")
        output = output + bias
        # output += 0.00052
        output = output * self.s_iwr['inverted_residual_1_projection'] + 128
        
        output_uint8 = output.round()
        output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
        add_1 = output_uint8.copy()
        # print()

        ################## inverted residual 1 add ##################
        add_1 = add_1.astype(np.float32)
        add_2 = add_2.astype(np.float32)

        # add_1 = tf.constant(0.1732778549194336, tf.float32) * (add_1 - 128)
        # add_2 = tf.constant(0.20100615918636322, tf.float32) * (add_2 - 128)
        add_1 = self.s_add['inverted_residual_1_add'][0] * (add_1 - 128)
        add_2 = self.s_add['inverted_residual_1_add'][1] * (add_2 - 128)

        output_result = add_1 + add_2
        # output = output_result / tf.constant(0.26455792784690857, tf.float32) + 128
        output = output_result / self.s_add['inverted_residual_1_add'][2] + 128
        output_uint8 = output.round()
        output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)

        ################## inverted residual 2 expansion ##################
        # print('inverted residual 2 expansion')
        new_data = output_uint8.astype(np.float32)
        new_data -= 128

        weight = self.weight['inverted_residual_2_expansion'].astype(np.float32)
        weight = weight - 128.
        weight = weight.transpose(1,2,3,0)
        # print(weight)

        bias = self.bias['inverted_residual_2_expansion'].astype(np.float32)
        # print(bias)

        output = conv2d(new_data, weight, stride=(1,1), pad="SAME")
        output = output + bias
        # output += 0.01062
        output = output * self.s_iwr['inverted_residual_2_expansion']

        output = relu(output)
        output += 128
        output_uint8 = output.round()
        output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
        # print()

        ################## inverted residual 2 depthwise ##################
        # print('inverted residual 2 depthwise')
        new_data = output_uint8.astype(np.float32)
        new_data -= 128

        weight = self.weight['inverted_residual_2_depthwise'].astype(np.float32)
        weight = weight - 128.
        weight = weight.transpose(1,2,3,0)
        # print(weight)

        bias = self.bias['inverted_residual_2_depthwise'].astype(np.float32)
        # print(bias)

        output = depthwise_conv2d(new_data, weight, stride=(1,1), pad="SAME")
        output = output + bias
        # output += 0.0153
        output = output * self.s_iwr['inverted_residual_2_depthwise']
        
        output = relu(output)
        output += 128
        output_uint8 = output.round()
        output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
        # print()

        ################## inverted residual 2 projection ##################
        # print('inverted residual 2 projection')
        new_data = output_uint8.astype(np.float32)
        new_data -= 128

        weight = self.weight['inverted_residual_2_projection'].astype(np.float32)
        weight = weight - 128.
        weight = weight.transpose(1,2,3,0)
        # print(weight)

        bias = self.bias['inverted_residual_2_projection'].astype(np.float32)
        # print(bias)

        output = conv2d(new_data, weight, stride=(1,1), pad="SAME")
        output = output + bias
        output = output * self.s_iwr['inverted_residual_2_projection'] + 128
        
        output_uint8 = output.round()
        output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
        add_2 = output_uint8.copy()
        # print()

        ################## inverted residual 3 expansion ##################
        # print('inverted residual 3 expansion')
        new_data = output_uint8.astype(np.float32)
        new_data -= 128

        weight = self.weight['inverted_residual_3_expansion'].astype(np.float32)
        weight = weight - 128.
        weight = weight.transpose(1,2,3,0)
        # print(weight)

        bias = self.bias['inverted_residual_3_expansion'].astype(np.float32)
        # print(bias)

        output = conv2d(new_data, weight, stride=(1,1), pad="SAME")
        output = output + bias
        # output += 0.00113
        output = output * self.s_iwr['inverted_residual_3_expansion']
        
        output = relu(output)
        output += 128
        output_uint8 = output.round()
        output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
        # print()

        ################## inverted residual 3 depthwise ##################
        # print('inverted residual 3 depthwise')
        new_data = output_uint8.astype(np.float32)
        new_data -= 128

        weight = self.weight['inverted_residual_3_depthwise'].astype(np.float32)
        weight = weight - 128.
        weight = weight.transpose(1,2,3,0)
        # print(weight)

        bias = self.bias['inverted_residual_3_depthwise'].astype(np.float32)
        # print(bias)

        output = depthwise_conv2d(new_data, weight, stride=(1,1), pad="SAME")
        output = output + bias
        output = output * self.s_iwr['inverted_residual_3_depthwise']
        
        output = relu(output)
        output += 128
        output_uint8 = output.round()
        output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
        # print()

        ################## inverted residual 3 projection ##################
        # print('inverted residual 3 projection')
        new_data = output_uint8.astype(np.float32)
        new_data -= 128

        weight = self.weight['inverted_residual_3_projection'].astype(np.float32)
        weight = weight - 128.
        weight = weight.transpose(1,2,3,0)
        # print(weight)

        bias = self.bias['inverted_residual_3_projection'].astype(np.float32)
        # print(bias)

        output = conv2d(new_data, weight, stride=(1,1), pad="SAME")
        output = output + bias
        output = output * self.s_iwr['inverted_residual_3_projection'] + 128
        
        output_uint8 = output.round()
        output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
        add_1 = output_uint8.copy()
        # print()

        ################## inverted residual 3 add ##################
        add_1 = add_1.astype(np.float32)
        add_2 = add_2.astype(np.float32)

        # add_1 = tf.constant(0.19232141971588135, tf.float32) * (add_1 - 128)
        # add_2 = tf.constant(0.12740808725357056, tf.float32) * (add_2 - 128)
        add_1 = self.s_add['inverted_residual_3_add'][0] * (add_1 - 128)
        add_2 = self.s_add['inverted_residual_3_add'][1] * (add_2 - 128)

        output_result = add_1 + add_2
        # output_uint8 = output_result / tf.constant(0.20970593392848969, tf.float32) + 128
        output = output_result / self.s_add['inverted_residual_3_add'][2] + 128
        output_uint8 = output.round()
        output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)

        ################## AvgPool ##################
        # method 1
        # new_data = tf.cast(output_uint8, tf.float32)
        # new_data = 0.21021947264671326 * (new_data - 131)
        # output = tf.nn.avg_pool(new_data,
        #                 ksize=[1,25,5,1],
        #                 strides=[1,25,5,1],
        #                 padding='VALID')
        # output = output / 0.21021947264671326 + 131
        # output_uint8 = tf.math.round(output)
        # output_uint8 = tf.cast(output, tf.uint8)

        # method 2 (简化版本，发现scale和zero_point完全可以消除)
        new_data = output_uint8.astype(np.float32)
        output = pooling(new_data, ksize=(25,5), method='mean')
        # output -= 0.0041
        output_uint8 = output.round()
        output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)

        ################## Conv2D ##################
        # print('Conv2D')
        new_data = output_uint8.astype(np.float32)
        new_data -= 128

        weight = self.weight['Conv2D'].astype(np.float32)
        weight = weight - 128.
        weight = weight.transpose(1,2,3,0)
        # print(weight)

        bias = self.bias['Conv2D'].astype(np.float32)
        # print(bias)

        output = conv2d(new_data, weight, stride=(1,1), pad="SAME")
        output = output + bias
        output = output * self.s_iwr['Conv2D'] + 128
        
        output_uint8 = output.round()
        output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
        # print()

        ################## Reshape ##################
        output_uint8 = np.squeeze(output_uint8, axis=(1,2))

        # ################## Softmax ##################
        # new_data = tf.cast(output_uint8, tf.float32)
        # new_data = tf.constant(0.1784215271472931, tf.float32) * (new_data - 129)
        # output = tf.nn.softmax(new_data)
        # output = output / tf.constant(0.00390625, tf.float32)

        # output_uint8 = tf.math.round(output)
        # output_uint8 = tf.cast(output_uint8, tf.uint8)

        ################## running ##################
        # return np.mean(np.equal(np.argmax(output_uint8, axis=1), np.argmax(label, axis=1)).astype(np.float32))
        return output_uint8

net = Net()

class Recoder:
    def __init__(self):
        self.NUM_SAMPLES = 2000       # pyaudio内置缓冲大小
        self.SAMPLING_RATE = 16000    # 取样频率
        self.LEVEL = 700              # 声音保存的阈值
        self.COUNT_NUM = 20           # NUM_SAMPLES个取样之内出现COUNT_NUM个大于LEVEL的取样则记录声音
        self.SAVE_LENGTH = 8          # 声音记录次数，每次记录1个缓冲区大小，共获得SAVE_LENGTH * NUM_SAMPLES 个取样
        self.BUFFER_BLOCK = 2         # 缓冲区，用于存储检测到输入之前的块，防止一些轻读被忽略，如stop的s

        self.FORMAT = paInt16
        self.CHANNELS = 1
        
        self.pa = PyAudio()
        self.Voice_String = []

    def __del__(self):
        self.pa.terminate()

    def savewav(self, filename):
        wf = wave.open(filename, 'wb') 
        wf.setnchannels(self.CHANNELS) 
        wf.setsampwidth(self.pa.get_sample_size(self.FORMAT))
        wf.setframerate(self.SAMPLING_RATE) 
        wf.writeframes(np.array(self.Voice_String).tobytes()) 
        # wf.writeframes(self.Voice_String.decode())
        wf.close()

    def recoding(self, inference_func):
        stream = self.pa.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.SAMPLING_RATE, input=True, 
            frames_per_buffer=self.NUM_SAMPLES) 
        save_flag = False
        save_buffer = []
        buffer_block = [np.zeros((1, self.NUM_SAMPLES), np.int16).tobytes() for _ in range(self.BUFFER_BLOCK)]   # 三块缓冲区，用于存储检测到输入之前的块，防止一些轻读被忽略，如stop的s
        print('Start recoding...')

        while True:
            try:
                # 读入NUM_SAMPLES个取样
                string_audio_data = stream.read(self.NUM_SAMPLES)   # len(string_audio_data) = 2048, 因为采样16bit，我们需要1024个16bit，因此这里有2048byte
                # 存储当前的数据，并删除距离当前最远的缓冲数据
                buffer_block.append(string_audio_data)
                buffer_block.pop(0)
                # 将读入的数据转换为数组
                audio_data = np.fromstring(string_audio_data, dtype=np.int16)
                # 计算大于LEVEL的取样的个数
                large_sample_count = np.sum(audio_data > self.LEVEL)
                print(np.max(audio_data))
                # 如果个数大于COUNT_NUM，则保存SAVE_LENGTH个块
                if large_sample_count > self.COUNT_NUM:
                    save_flag = True

                if save_flag:
                    save_buffer.append(string_audio_data)
                    if len(save_buffer) == self.SAVE_LENGTH - len(buffer_block):
                        self.Voice_String = buffer_block + save_buffer
                        print('Recognize one command')

                        inference_func(self.Voice_String)

                        save_buffer = []
                        save_flag = False

            except KeyboardInterrupt:
                print('Stop recording')
                soct.send('###')
                break
        stream.stop_stream()
        stream.close()

def run_inference(wav_data):

    words_list = "silence,unknown,yes,no,up,down,left,right,on,off,stop,go".split(',')
    # print(words_list)

    ########################### inference ###########################
    start_time = time.time()

    fingerprints = psf_mfcc(wav_data).flatten()
    fingerprints = fp32_to_uint8(fingerprints)
    output_uint8 = net(fingerprints.reshape(-1, 49, 10, 1))
    predicted_indices = np.argmax(output_uint8, 1)

    end_time = time.time()
    print('Running time: {} second'.format(end_time - start_time))
    
    soct.send(words_list[predicted_indices[0]])
    print(words_list[predicted_indices[0]])

if __name__ == '__main__':
    r = Recoder()
    r.recoding(inference_func=run_inference)