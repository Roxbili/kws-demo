#-*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import numpy as np
import time
import socket

from layers import conv2d, depthwise_conv2d, relu, pooling
import input_data_zynq as input_data
import models_zynq as models
from input_data_zynq import psf_mfcc

# os.chdir('../')

class Soct(object):
    def __init__(self, address):
        # buffer_size = 16
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.s.connect(address)  # 尝试连接服务端
        except Exception:
            print('[!] Server not found ot not open')
            sys.exit(0)
    
    def __del__(self):
        self.s.close()

    def send(self, info):
        self.s.sendall(info.encode())

address = ('192.168.2.151', 6887)  # 服务端地址和端口
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

def run_inference(args):
    """Creates an audio model with the nodes needed for inference.

    Uses the supplied arguments to create a model, and inserts the input and
    output nodes that are needed to use the graph for inference.
    """


    words_list = ("silence,unknown," + args.wanted_words).split(',')
    # print(words_list)

    ########################### inference ###########################
    # wav_path = '/share/Downloads/speech/yes/15dd287d_nohash_3.wav'
    # wav_path = '/share/Downloads/speech/left/0a2b400e_nohash_0.wav'
    wav_path = 'speech_dataset/left/ffd2ba2f_nohash_2.wav'

    while True:
        start_time = time.time()

        fingerprints = psf_mfcc(wav_path).flatten()
        fingerprints = fp32_to_uint8(fingerprints)
        output_uint8 = net(fingerprints.reshape(-1, 49, 10, 1))
        predicted_indices = np.argmax(output_uint8, 1)
        # print(words_list[predicted_indices[0]])

        ########################### simulate lite model ###########################
        end_time = time.time()
        print('Running time: {} second'.format(end_time - start_time))
        
        soct.send(words_list[predicted_indices[0]])

        time.sleep(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_url',
        type=str,
        # pylint: disable=line-too-long
        default='http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz',
        # pylint: enable=line-too-long
        help='Location of speech training data archive on the web.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/speech_dataset/',
        help="""\
            Where to download the speech training data to.
            """)
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
            How much of the training data should be silence.
            """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=10.0,
        help="""\
            How much of the training data should be unknown words.
            """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of wavs to use as a validation set.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs',)
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs',)
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint',)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='How many items to train with at once',)
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)',)
    parser.add_argument(
        '--tflite_path',
        type=str,
        default='',
        help='The path where tflite model saved.')
    parser.add_argument(
        '--model_architecture',
        type=str,
        default='dnn',
        help='What model architecture to use')
    parser.add_argument(
        '--model_size_info',
        type=int,
        nargs="+",
        default=[128,128,128],
        help='Model dimensions - different for various models')
    parser.add_argument(
        '--testing_mode',
        type=str,
        default="real",
        help='real | simulate')

    args = parser.parse_args()
    run_inference(args)
