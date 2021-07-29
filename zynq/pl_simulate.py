#-*- encoding: utf-8 -*-

import numpy as np
import os, sys
from multiprocessing import Process, Queue

from kws_ps_pl import BRAM
from test_tflite_numpy import simulate_net
from layers import conv2d, depthwise_conv2d, relu, pooling

def simulate_net(input_data):
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
    
    # model_dir = 'test_log/mobilenetv3_quant_gen'
    model_dir = 'test_log/mobilenetv3_quant_mfcc_gen'

    ################## stem conv ##################
    # print('stem conv')
    new_data = input_data.astype(np.float32)
    new_data = new_data - 221.
    # s_iwr = tf.constant(0.0008852639002725482 / 0.20100615918636322, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_stem_conv_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_stem_conv_conv_Conv2D_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = weight.astype(np.float32)
    weight = weight - 128.
    weight = weight.transpose(1,2,0,3)

    bias = bias.astype(np.float32)
    # print(weight)
    # print(bias)

    output = depthwise_conv2d(new_data, weight, stride=(2,2), pad="SAME")

    output += bias
    output = output * s_iwr['stem_conv']
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
    # s_iwr = tf.constant(0.0035931775346398354 / 0.42823609709739685, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_expansion_conv_Conv2D_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = weight.astype(np.float32)
    weight = weight - 128.
    weight = weight.transpose(1,2,3,0)
    # print(weight)

    bias = bias.astype(np.float32)
    # print(bias)

    output = conv2d(new_data, weight, stride=(1,1), pad="SAME")
    output = output + bias
    # output += 0.0074
    output = output * s_iwr['inverted_residual_1_expansion']
    
    output = relu(output)
    output += 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    # print()

    ################## inverted residual 1 depthwise ##################
    # print('inverted residual 1 depthwise')
    new_data = output_uint8.astype(np.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.00785899069160223 / 0.23841151595115662, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_depthwise_depthwise_conv_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = weight.astype(np.float32)
    weight = weight - 128.
    weight = weight.transpose(1,2,3,0)
    # print(weight)

    bias = bias.astype(np.float32)
    # print(bias)

    output = depthwise_conv2d(new_data, weight, stride=(1,1), pad="SAME")
    output = output + bias
    # output += 0.0301
    output = output * s_iwr['inverted_residual_1_depthwise']
    
    output = relu(output)
    output += 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    # print()

    ################## inverted residual 1 projection ##################
    # print('inverted residual 1 projection')
    new_data = output_uint8.astype(np.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.0014689048985019326 / 0.1732778549194336, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_projection_conv_Conv2D_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = weight.astype(np.float32)
    weight = weight - 128.
    weight = weight.transpose(1,2,3,0)
    # print(weight)

    bias = bias.astype(np.float32)
    # print(bias)

    output = conv2d(new_data, weight, stride=(1,1), pad="SAME")
    output = output + bias
    # output += 0.00052
    output = output * s_iwr['inverted_residual_1_projection'] + 128
    
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    add_1 = output_uint8.copy()
    # print()

    ################## inverted residual 1 add ##################
    add_1 = add_1.astype(np.float32)
    add_2 = add_2.astype(np.float32)

    # add_1 = tf.constant(0.1732778549194336, tf.float32) * (add_1 - 128)
    # add_2 = tf.constant(0.20100615918636322, tf.float32) * (add_2 - 128)
    add_1 = s_add['inverted_residual_1_add'][0] * (add_1 - 128)
    add_2 = s_add['inverted_residual_1_add'][1] * (add_2 - 128)

    output_result = add_1 + add_2
    # output = output_result / tf.constant(0.26455792784690857, tf.float32) + 128
    output = output_result / s_add['inverted_residual_1_add'][2] + 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)

    ################## inverted residual 2 expansion ##################
    # print('inverted residual 2 expansion')
    new_data = output_uint8.astype(np.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.0015524440677836537 / 0.21222199499607086, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_expansion_conv_Conv2D_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = weight.astype(np.float32)
    weight = weight - 128.
    weight = weight.transpose(1,2,3,0)
    # print(weight)

    bias = bias.astype(np.float32)
    # print(bias)

    output = conv2d(new_data, weight, stride=(1,1), pad="SAME")
    output = output + bias
    # output += 0.01062
    output = output * s_iwr['inverted_residual_2_expansion']

    output = relu(output)
    output += 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    # print()

    ################## inverted residual 2 depthwise ##################
    # print('inverted residual 2 depthwise')
    new_data = output_uint8.astype(np.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.0028435662388801575 / 0.15781369805335999, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_depthwise_depthwise_conv_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = weight.astype(np.float32)
    weight = weight - 128.
    weight = weight.transpose(1,2,3,0)
    # print(weight)

    bias = bias.astype(np.float32)
    # print(bias)

    output = depthwise_conv2d(new_data, weight, stride=(1,1), pad="SAME")
    output = output + bias
    # output += 0.0153
    output = output * s_iwr['inverted_residual_2_depthwise']
    
    output = relu(output)
    output += 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    # print()

    ################## inverted residual 2 projection ##################
    # print('inverted residual 2 projection')
    new_data = output_uint8.astype(np.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.001141879241913557 / 0.12740808725357056, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_projection_conv_Conv2D_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = weight.astype(np.float32)
    weight = weight - 128.
    weight = weight.transpose(1,2,3,0)
    # print(weight)

    bias = bias.astype(np.float32)
    # print(bias)

    output = conv2d(new_data, weight, stride=(1,1), pad="SAME")
    output = output + bias
    output = output * s_iwr['inverted_residual_2_projection'] + 128
    
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    add_2 = output_uint8.copy()
    # print()

    ################## inverted residual 3 expansion ##################
    # print('inverted residual 3 expansion')
    new_data = output_uint8.astype(np.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.0007087105768732727 / 0.1111915186047554, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_expansion_conv_Conv2D_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = weight.astype(np.float32)
    weight = weight - 128.
    weight = weight.transpose(1,2,3,0)
    # print(weight)

    bias = bias.astype(np.float32)
    # print(bias)

    output = conv2d(new_data, weight, stride=(1,1), pad="SAME")
    output = output + bias
    # output += 0.00113
    output = output * s_iwr['inverted_residual_3_expansion']
    
    output = relu(output)
    output += 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    # print()

    ################## inverted residual 3 depthwise ##################
    # print('inverted residual 3 depthwise')
    new_data = output_uint8.astype(np.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.009289528243243694 / 0.11338130384683609, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_depthwise_depthwise_conv_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = weight.astype(np.float32)
    weight = weight - 128.
    weight = weight.transpose(1,2,3,0)
    # print(weight)

    bias = bias.astype(np.float32)
    # print(bias)

    output = depthwise_conv2d(new_data, weight, stride=(1,1), pad="SAME")
    output = output + bias
    output = output * s_iwr['inverted_residual_3_depthwise']
    
    output = relu(output)
    output += 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    # print()

    ################## inverted residual 3 projection ##################
    # print('inverted residual 3 projection')
    new_data = output_uint8.astype(np.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.0015117411967366934 / 0.19232141971588135, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_projection_conv_Conv2D_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = weight.astype(np.float32)
    weight = weight - 128.
    weight = weight.transpose(1,2,3,0)
    # print(weight)

    bias = bias.astype(np.float32)
    # print(bias)

    output = conv2d(new_data, weight, stride=(1,1), pad="SAME")
    output = output + bias
    output = output * s_iwr['inverted_residual_3_projection'] + 128
    
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    add_1 = output_uint8.copy()
    # print()

    ################## inverted residual 3 add ##################
    add_1 = add_1.astype(np.float32)
    add_2 = add_2.astype(np.float32)

    # add_1 = tf.constant(0.19232141971588135, tf.float32) * (add_1 - 128)
    # add_2 = tf.constant(0.12740808725357056, tf.float32) * (add_2 - 128)
    add_1 = s_add['inverted_residual_3_add'][0] * (add_1 - 128)
    add_2 = s_add['inverted_residual_3_add'][1] * (add_2 - 128)

    output_result = add_1 + add_2
    # output_uint8 = output_result / tf.constant(0.20970593392848969, tf.float32) + 128
    output = output_result / s_add['inverted_residual_3_add'][2] + 128
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
    # s_iwr = tf.constant(0.004092711955308914 / 0.17540767788887024, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_fc_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_fc_conv_Conv2D_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = weight.astype(np.float32)
    weight = weight - 128.
    weight = weight.transpose(1,2,3,0)
    # print(weight)

    bias = bias.astype(np.float32)
    # print(bias)

    output = conv2d(new_data, weight, stride=(1,1), pad="SAME")
    output = output + bias
    output = output * s_iwr['Conv2D'] + 128
    
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

class PL(object):
    def __init__(self):
        self.bram = BRAM()
        self.q = Queue()

    def get_data_to_queue(self):
        for _ in range(100):    # 先保存100张图片到队列里面，毕竟这只是个模拟，实际未必这么操作
            self.bram.write(b'\x01\x00\x00\x00', start=0x4, map_id=1)
            # input flag == 1?
            while True:
                input_flag = self.bram.read_oneByOne(1, start=0x1)
                if input_flag[0] == 1:
                    self.bram.write(b'\x01\x00\x00\x00', start=0x0)
                    break
            # read data
            data = self.bram.read_oneByOne(490, start=0x30C0)
            self.q.put(data)
        
    def inference(self):
        while True:
            data = self.q.get(block=True, timeout=5)
            output_uint8 = simulate_net(data.reshape(-1, 49, 10, 1))
            # predicted_indices = np.argmax(output_uint8, 1)

            self.bram.write(output_uint8, start=0x4, map_id=1)
            # result flag
            self.bram.write(b'\x01\x00\x00\x00', start=0x0, map_id=1)

if __name__ == '__main__':
    pl = PL()
    p1 = Process(target=pl.get_data_to_queue)
    p2 = Process(target=pl.inference)

    print('Start pl simulating...')
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    print('Done')