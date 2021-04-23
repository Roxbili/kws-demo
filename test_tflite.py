#-*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import numpy as np
import tensorflow as tf
import time

import input_data
import models
from gen_bin import save_bin


def data_stats(train_data, val_data, test_data):
    """mean and std_dev

        Args:
            train_data: (36923, 490)
            val_data: (4445, 490)
            test_data: (4890, 490)

        Return: (mean, std_dev)

        Result:
            mean: -3.975149608704592, 220.81257374779565
            std_dev: 0.8934739293234528
    """
    print(train_data.shape, val_data.shape, test_data.shape)
    all_data = np.concatenate((train_data, val_data, test_data), axis=0)
    std_dev = 255. / (all_data.max() - all_data.min())
    # mean_ = all_data.mean()
    mean_ = 255. * all_data.min() / (all_data.min() - all_data.max())
    return (mean_, std_dev)

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

def simulate_net(input_data):
    sess = tf.Session()
    # tf mfcc parameters
    # bias_scale = np.array([0.0008852639002725482, 0.0035931775346398354, 0.00785899069160223, 0.0014689048985019326, 0.0015524440677836537, 0.0028435662388801575, 0.001141879241913557, 0.0007087105768732727, 0.009289528243243694, 0.0015117411967366934, 0.004092711955308914])
    # result_sacale = np.array([0.20100615918636322, 0.42823609709739685, 0.23841151595115662, 0.1732778549194336, 0.21222199499607086, 0.15781369805335999, 0.12740808725357056, 0.1111915186047554, 0.11338130384683609, 0.19232141971588135, 0.17540767788887024])
    # add_scale = np.array([0.1732778549194336, 0.20100615918636322, 0.26455792784690857, 0.19232141971588135, 0.12740808725357056, 0.20970593392848969])

    # new mfcc parameters
    bias_scale = np.array([0.0005171183147467673, 0.0021205246448516846, 0.004102946724742651, 0.0007573990151286125, 0.0009573157876729965, 0.0045410459861159325, 0.0007452332065440714, 0.0003749248862732202, 0.0028607698623090982, 0.0014322539791464806, 0.0036672416608780622])
    result_sacale = np.array([0.12663139402866364, 0.20024137198925018, 0.13141511380672455, 0.11106141656637192, 0.1328522115945816, 0.08316611498594284, 0.08792730420827866, 0.08202825486660004, 0.1061563566327095, 0.17049182951450348, 0.18540261685848236])
    add_scale = np.array([0.11106141656637192, 0.12663139402866364, 0.13807182013988495, 0.17049182951450348, 0.08792730420827866, 0.20207594335079193])

    scale = bias_scale / result_sacale
    scale = tf.convert_to_tensor(np.round(scale * 2**10) / 2**10, tf.float32)
    add_scale = tf.convert_to_tensor(np.round(add_scale * 2**10) / 2**10, tf.float32)

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
    new_data = tf.cast(input_data, tf.float32)
    new_data = new_data - 221.
    # s_iwr = tf.constant(0.0008852639002725482 / 0.20100615918636322, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_stem_conv_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_stem_conv_conv_Conv2D_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32, name='weight')
    weight = weight - 128.
    weight = tf.transpose(weight, perm=[1,2,0,3])

    bias = tf.convert_to_tensor(bias, tf.float32, name='bias')
    # print(weight)
    # print(bias)

    output = tf.nn.depthwise_conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,2,2,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None,  # 数据格式，与步长参数配合，决定移动方式
                name='stem_conv') # 名字，用于tensorboard图形显示时使用

    output = tf.add(output, bias, name='add')
    output = output * s_iwr['stem_conv']
    # output += 0.0035
    
    output = tf.nn.relu(output)
    output += 128
    output_uint8 = tf.math.round(output, name='round')
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8, name='uint8')
    add_2 = tf.identity(output_uint8)   # 给之后的做加法
    # print()

    ################## inverted residual 1 expansion ##################
    # print('inverted residual 1 expansion')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.0035931775346398354 / 0.42823609709739685, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_expansion_conv_Conv2D_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    # output += 0.0074
    output = output * s_iwr['inverted_residual_1_expansion']
    
    output = tf.nn.relu(output)
    output += 128
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()

    ################## inverted residual 1 depthwise ##################
    # print('inverted residual 1 depthwise')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.00785899069160223 / 0.23841151595115662, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_depthwise_depthwise_conv_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.depthwise_conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    # output += 0.0301
    output = output * s_iwr['inverted_residual_1_depthwise']
    
    output = tf.nn.relu(output)
    output += 128
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()

    ################## inverted residual 1 projection ##################
    # print('inverted residual 1 projection')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.0014689048985019326 / 0.1732778549194336, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_1_projection_conv_Conv2D_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    # output += 0.00052
    output = output * s_iwr['inverted_residual_1_projection'] + 128
    
    output_uint8 = tf.math.round(output)
    mask1 = tf.ones_like(output_uint8) * 255
    mask2 = tf.zeros_like(output_uint8)
    output_uint8 = tf.where(output_uint8 > 255, mask1, output_uint8)
    output_uint8 = tf.where(output_uint8 < 0, mask2, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    add_1 = tf.identity(output_uint8)
    # print()

    ################## inverted residual 1 add ##################
    add_1 = tf.cast(add_1, tf.float32)
    add_2 = tf.cast(add_2, tf.float32)

    # add_1 = tf.constant(0.1732778549194336, tf.float32) * (add_1 - 128)
    # add_2 = tf.constant(0.20100615918636322, tf.float32) * (add_2 - 128)
    add_1 = s_add['inverted_residual_1_add'][0] * (add_1 - 128)
    add_2 = s_add['inverted_residual_1_add'][1] * (add_2 - 128)

    output_result = tf.add(add_1, add_2)
    # output = output_result / tf.constant(0.26455792784690857, tf.float32) + 128
    output = output_result / s_add['inverted_residual_1_add'][2] + 128
    output_uint8 = tf.math.round(output)
    mask1 = tf.ones_like(output_uint8) * 255
    mask2 = tf.zeros_like(output_uint8)
    output_uint8 = tf.where(output_uint8 > 255, mask1, output_uint8)
    output_uint8 = tf.where(output_uint8 < 0, mask2, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)

    ################## inverted residual 2 expansion ##################
    # print('inverted residual 2 expansion')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.0015524440677836537 / 0.21222199499607086, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_expansion_conv_Conv2D_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    # output += 0.01062
    output = output * s_iwr['inverted_residual_2_expansion']

    output = tf.nn.relu(output)
    output += 128
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()

    ################## inverted residual 2 depthwise ##################
    # print('inverted residual 2 depthwise')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.0028435662388801575 / 0.15781369805335999, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_depthwise_depthwise_conv_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.depthwise_conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    # output += 0.0153
    output = output * s_iwr['inverted_residual_2_depthwise']
    
    output = tf.nn.relu(output)
    output += 128
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()

    ################## inverted residual 2 projection ##################
    # print('inverted residual 2 projection')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.001141879241913557 / 0.12740808725357056, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_2_projection_conv_Conv2D_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    output = output * s_iwr['inverted_residual_2_projection'] + 128
    
    output_uint8 = tf.math.round(output)
    mask1 = tf.ones_like(output_uint8) * 255
    mask2 = tf.zeros_like(output_uint8)
    output_uint8 = tf.where(output_uint8 > 255, mask1, output_uint8)
    output_uint8 = tf.where(output_uint8 < 0, mask2, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    add_2 = tf.identity(output_uint8)
    # print()

    ################## inverted residual 3 expansion ##################
    # print('inverted residual 3 expansion')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.0007087105768732727 / 0.1111915186047554, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_expansion_conv_Conv2D_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    # output += 0.00113
    output = output * s_iwr['inverted_residual_3_expansion']
    
    output = tf.nn.relu(output)
    output += 128
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()

    ################## inverted residual 3 depthwise ##################
    # print('inverted residual 3 depthwise')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.009289528243243694 / 0.11338130384683609, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_depthwise_depthwise_conv_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.depthwise_conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    output = output * s_iwr['inverted_residual_3_depthwise']
    
    output = tf.nn.relu(output)
    output += 128
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()

    ################## inverted residual 3 projection ##################
    # print('inverted residual 3 projection')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.0015117411967366934 / 0.19232141971588135, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_inverted_residual_3_projection_conv_Conv2D_Fold_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    output = output * s_iwr['inverted_residual_3_projection'] + 128
    
    output_uint8 = tf.math.round(output)
    mask1 = tf.ones_like(output_uint8) * 255
    mask2 = tf.zeros_like(output_uint8)
    output_uint8 = tf.where(output_uint8 > 255, mask1, output_uint8)
    output_uint8 = tf.where(output_uint8 < 0, mask2, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    add_1 = tf.identity(output_uint8)
    # print()

    ################## inverted residual 3 add ##################
    add_1 = tf.cast(add_1, tf.float32)
    add_2 = tf.cast(add_2, tf.float32)

    # add_1 = tf.constant(0.19232141971588135, tf.float32) * (add_1 - 128)
    # add_2 = tf.constant(0.12740808725357056, tf.float32) * (add_2 - 128)
    add_1 = s_add['inverted_residual_3_add'][0] * (add_1 - 128)
    add_2 = s_add['inverted_residual_3_add'][1] * (add_2 - 128)

    output_result = tf.add(add_1, add_2)
    # output_uint8 = output_result / tf.constant(0.20970593392848969, tf.float32) + 128
    output_uint8 = output_result / s_add['inverted_residual_3_add'][2] + 128
    output_uint8 = tf.math.round(output_uint8)
    mask1 = tf.ones_like(output_uint8) * 255
    mask2 = tf.zeros_like(output_uint8)
    output_uint8 = tf.where(output_uint8 > 255, mask1, output_uint8)
    output_uint8 = tf.where(output_uint8 < 0, mask2, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)

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
    new_data = tf.cast(output_uint8, tf.float32)
    output = tf.nn.avg_pool(new_data,
                    ksize=[1,25,5,1],
                    strides=[1,25,5,1],
                    padding='VALID')
    # output -= 0.0041
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)

    ################## Conv2D ##################
    # print('Conv2D')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    # s_iwr = tf.constant(0.004092711955308914 / 0.17540767788887024, tf.float32)
    # s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_fc_conv_weights_quant_FakeQuantWithMinMaxVars.npy'))
    bias = np.load(os.path.join(model_dir, 'weight/MBNetV3-CNN_fc_conv_Conv2D_bias.npy'))
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    output = output * s_iwr['Conv2D'] + 128
    
    output_uint8 = tf.math.round(output)
    mask1 = tf.ones_like(output_uint8) * 255
    mask2 = tf.zeros_like(output_uint8)
    output_uint8 = tf.where(output_uint8 > 255, mask1, output_uint8)
    output_uint8 = tf.where(output_uint8 < 0, mask2, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    add_1 = tf.identity(output_uint8)
    # print()

    ################## Reshape ##################
    output_uint8 = tf.squeeze(output_uint8, axis=[1,2])

    # ################## Softmax ##################
    # new_data = tf.cast(output_uint8, tf.float32)
    # new_data = tf.constant(0.1784215271472931, tf.float32) * (new_data - 129)
    # output = tf.nn.softmax(new_data)
    # output = output / tf.constant(0.00390625, tf.float32)

    # output_uint8 = tf.math.round(output)
    # output_uint8 = tf.cast(output_uint8, tf.uint8)

    ################## running ##################
    # return np.mean(np.equal(np.argmax(output_uint8, axis=1), np.argmax(label, axis=1)).astype(np.float32))
    return (sess, output_uint8)

def calc(interpreter, input_data):

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details)
    # print(output_details)

    # Test model on random input data.
    # input_shape = input_details[0]['shape']
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # print(output_data)
    # print(label)

    # return np.mean(np.equal(np.argmax(output_data, axis=1), np.argmax(label, axis=1)).astype(np.float32))
    return output_data

def debug(output_uint8, add_2):
    '''输入正确的结果，检查后面的网络'''
    sess = tf.Session()
    ################## stem conv ##################
    # print('stem conv')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data = new_data - 221.
    s_iwr = tf.constant(0.0008852639002725482 / 0.20100615918636322, tf.float32)
    s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_stem_conv_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_stem_conv_conv_Conv2D_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128.
    weight = tf.transpose(weight, perm=[1,2,0,3])

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(weight)
    # print(bias)

    output = tf.nn.depthwise_conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,2,2,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式

    output = tf.add(output, bias)
    output = output * s_iwr
    # output += 0.0035
    
    output = tf.nn.relu(output)
    output += 128
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    add_2 = tf.identity(output_uint8)   # 给之后的做加法
    # print()
    ################## inverted residual 1 expansion ##################
    # print('inverted residual 1 expansion')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    s_iwr = tf.constant(0.0035931775346398354 / 0.42823609709739685, tf.float32)
    s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_1_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_1_expansion_conv_Conv2D_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    # output += 0.0074
    output = output * s_iwr
    
    output = tf.nn.relu(output)
    output += 128
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()
    ################## inverted residual 1 depthwise ##################
    # print('inverted residual 1 depthwise')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    s_iwr = tf.constant(0.00785899069160223 / 0.23841151595115662, tf.float32)
    s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_1_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_1_depthwise_depthwise_conv_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.depthwise_conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    # output += 0.0301
    output = output * s_iwr
    
    output = tf.nn.relu(output)
    output += 128
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()
    ################## inverted residual 1 projection ##################
    # print('inverted residual 1 projection')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    s_iwr = tf.constant(0.0014689048985019326 / 0.1732778549194336, tf.float32)
    s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_1_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_1_projection_conv_Conv2D_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    # output += 0.00052
    output = output * s_iwr + 128
    
    output_uint8 = tf.math.round(output)
    mask1 = tf.ones_like(output_uint8) * 255
    mask2 = tf.zeros_like(output_uint8)
    output_uint8 = tf.where(output_uint8 > 255, mask1, output_uint8)
    output_uint8 = tf.where(output_uint8 < 0, mask2, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    add_1 = tf.identity(output_uint8)
    # print()
    ################## inverted residual 1 add ##################
    add_1 = tf.cast(output_uint8, tf.float32)
    add_2 = tf.cast(add_2, tf.float32)

    add_1 = tf.constant(0.1732778549194336, tf.float32) * (add_1 - 128)
    add_2 = tf.constant(0.20100615918636322, tf.float32) * (add_2 - 128)

    output_result = tf.add(add_1, add_2)
    output = output_result / tf.constant(0.26455792784690857, tf.float32) + 128
    output_uint8 = tf.math.round(output)
    mask1 = tf.ones_like(output_uint8) * 255
    mask2 = tf.zeros_like(output_uint8)
    output_uint8 = tf.where(output_uint8 > 255, mask1, output_uint8)
    output_uint8 = tf.where(output_uint8 < 0, mask2, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    ################## inverted residual 2 expansion ##################
    # print('inverted residual 2 expansion')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    s_iwr = tf.constant(0.0015524440677836537 / 0.21222199499607086, tf.float32)
    s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_2_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_2_expansion_conv_Conv2D_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    # output += 0.01062
    output = output * s_iwr

    output = tf.nn.relu(output)
    output += 128
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()
    ################## inverted residual 2 depthwise ##################
    # print('inverted residual 2 depthwise')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    s_iwr = tf.constant(0.0028435662388801575 / 0.15781369805335999, tf.float32)
    s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_2_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_2_depthwise_depthwise_conv_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.depthwise_conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    # output += 0.0153
    output = output * s_iwr
    
    output = tf.nn.relu(output)
    output += 128
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()
    ################## inverted residual 2 projection ##################
    # print('inverted residual 2 projection')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    s_iwr = tf.constant(0.001141879241913557 / 0.12740808725357056, tf.float32)
    s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_2_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_2_projection_conv_Conv2D_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    output = output * s_iwr + 128
    
    output_uint8 = tf.math.round(output)
    mask1 = tf.ones_like(output_uint8) * 255
    mask2 = tf.zeros_like(output_uint8)
    output_uint8 = tf.where(output_uint8 > 255, mask1, output_uint8)
    output_uint8 = tf.where(output_uint8 < 0, mask2, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    add_2 = tf.identity(output_uint8)
    # print()
    ################## inverted residual 3 expansion ##################
    # print('inverted residual 3 expansion')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    s_iwr = tf.constant(0.0007087105768732727 / 0.1111915186047554, tf.float32)
    s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_3_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_3_expansion_conv_Conv2D_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    # output += 0.00113
    output = output * s_iwr
    
    output = tf.nn.relu(output)
    output += 128
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()
    ################## inverted residual 3 depthwise ##################
    # print('inverted residual 3 depthwise')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    s_iwr = tf.constant(0.009289528243243694 / 0.11338130384683609, tf.float32)
    s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_3_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_3_depthwise_depthwise_conv_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.depthwise_conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    output = output * s_iwr
    
    output = tf.nn.relu(output)
    output += 128
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()
    ################## inverted residual 3 projection ##################
    # print('inverted residual 3 projection')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    s_iwr = tf.constant(0.0015117411967366934 / 0.19232141971588135, tf.float32)
    s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_3_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_inverted_residual_3_projection_conv_Conv2D_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    output = output * s_iwr + 128
    
    output_uint8 = tf.math.round(output)
    mask1 = tf.ones_like(output_uint8) * 255
    mask2 = tf.zeros_like(output_uint8)
    output_uint8 = tf.where(output_uint8 > 255, mask1, output_uint8)
    output_uint8 = tf.where(output_uint8 < 0, mask2, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    add_1 = tf.identity(output_uint8)
    # print()
    ################## inverted residual 3 add ##################
    add_1 = tf.cast(output_uint8, tf.float32)
    add_2 = tf.cast(add_2, tf.float32)

    add_1 = tf.constant(0.19232141971588135, tf.float32) * (add_1 - 128)
    add_2 = tf.constant(0.12740808725357056, tf.float32) * (add_2 - 128)

    output_result = tf.add(add_1, add_2)
    output_uint8 = output_result / tf.constant(0.20970593392848969, tf.float32) + 128
    output_uint8 = tf.math.round(output_uint8)
    mask1 = tf.ones_like(output_uint8) * 255
    mask2 = tf.zeros_like(output_uint8)
    output_uint8 = tf.where(output_uint8 > 255, mask1, output_uint8)
    output_uint8 = tf.where(output_uint8 < 0, mask2, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
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
    new_data = tf.cast(output_uint8, tf.float32)
    output = tf.nn.avg_pool(new_data,
                    ksize=[1,25,5,1],
                    strides=[1,25,5,1],
                    padding='VALID')
    # output -= 0.0041
    output_uint8 = tf.math.round(output)
    mask = tf.ones_like(output_uint8) * 255
    output_uint8 = tf.where(output_uint8 > 255, mask, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    ################## Conv2D ##################
    # print('Conv2D')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 128
    s_iwr = tf.constant(0.004092711955308914 / 0.17540767788887024, tf.float32)
    s_iwr = tf.cast(s_iwr, tf.float32)

    weight = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_fc_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_gen/weight/MBNetV3-CNN_fc_conv_Conv2D_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 128
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = output + bias
    output = output * s_iwr + 128
    
    output_uint8 = tf.math.round(output)
    mask1 = tf.ones_like(output_uint8) * 255
    mask2 = tf.zeros_like(output_uint8)
    output_uint8 = tf.where(output_uint8 > 255, mask1, output_uint8)
    output_uint8 = tf.where(output_uint8 < 0, mask2, output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    add_1 = tf.identity(output_uint8)
    # print()

    ################## Reshape ##################
    output_uint8 = tf.squeeze(output_uint8, axis=[1,2])

    # ################## Softmax ##################
    # new_data = tf.cast(output_uint8, tf.float32)
    # new_data = tf.constant(0.1784215271472931, tf.float32) * (new_data - 129)
    # output = tf.nn.softmax(new_data)
    # output = output / tf.constant(0.00390625, tf.float32)

    # output_uint8 = tf.math.round(output)
    # output_uint8 = tf.cast(output_uint8, tf.uint8)

    ################## running ##################
    # return np.mean(np.equal(np.argmax(output_uint8, axis=1), np.argmax(label, axis=1)).astype(np.float32))
    return (sess, output_uint8, new_data, output)

def run_inference(wanted_words, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms, dct_coefficient_count, 
                           model_architecture, model_size_info):
    """Creates an audio model with the nodes needed for inference.

    Uses the supplied arguments to create a model, and inserts the input and
    output nodes that are needed to use the graph for inference.

    Args:
        wanted_words: Comma-separated list of the words we're trying to recognize.
        sample_rate: How many samples per second are in the input audio files.
        clip_duration_ms: How many samples to analyze for the audio pattern.
        window_size_ms: Time slice duration to estimate frequencies from.
        window_stride_ms: How far apart time slices should be.
        dct_coefficient_count: Number of frequency bands to analyze.
        model_architecture: Name of the kind of model to generate.
        model_size_info: Model dimensions : different lengths for different models
    """

    
    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()
    words_list = input_data.prepare_words_list(wanted_words.split(','))
    model_settings = models.prepare_model_settings(
        len(words_list), sample_rate, clip_duration_ms, window_size_ms,
        window_stride_ms, dct_coefficient_count)

    audio_processor = input_data.AudioProcessor(
        FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings)
    
    label_count = model_settings['label_count']
    fingerprint_size = model_settings['fingerprint_size']

    # fingerprint_input = tf.placeholder(
    #     tf.float32, [None, fingerprint_size], name='fingerprint_input')

    # logits = models.create_model(
    #     fingerprint_input,
    #     model_settings,
    #     FLAGS.model_architecture,
    #     FLAGS.model_size_info,
    #     is_training=False)

    # ground_truth_input = tf.placeholder(
    #     tf.float32, [None, label_count], name='groundtruth_input')

    # predicted_indices = tf.argmax(logits, 1)
    # expected_indices = tf.argmax(ground_truth_input, 1)
    # correct_prediction = tf.equal(predicted_indices, expected_indices)
    # confusion_matrix = tf.confusion_matrix(
    #     expected_indices, predicted_indices, num_classes=label_count)
    # evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=FLAGS.tflite_path)
    # interpreter = tf.lite.Interpreter(model_path="tflite_factory/swiftnet-uint8.lite")
    interpreter.allocate_tensors()


    start_time = time.time()
    if FLAGS.testing_mode == 'real':
        ########################### lite model ###########################
        # training set
        set_size = audio_processor.set_size('training')
        tf.logging.info('set_size=%d', set_size)
        total_accuracy = 0
        for i in range(0, set_size, FLAGS.batch_size):
            training_fingerprints, training_ground_truth = audio_processor.get_data(
                FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'training', sess)
            # print(test_fingerprints.shape)  # (batch_size 490)
            # print(test_ground_truth.shape)  # (batch_size, 12)
            training_fingerprints = fp32_to_uint8(training_fingerprints)
            output_data = calc(interpreter, training_fingerprints)
            training_accuracy = np.mean(np.equal(np.argmax(output_data, axis=1), np.argmax(training_ground_truth, axis=1)).astype(np.float32))

            batch_size = min(FLAGS.batch_size, set_size - i)
            total_accuracy += (training_accuracy * batch_size) / set_size

        tf.logging.info('Training accuracy = %.2f%% (N=%d)' %
                        (total_accuracy * 100, set_size))

        # validation set
        set_size = audio_processor.set_size('validation')
        tf.logging.info('set_size=%d', set_size)
        total_accuracy = 0
        for i in range(0, set_size, FLAGS.batch_size):
            validation_fingerprints, validation_ground_truth = audio_processor.get_data(
                FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'validation', sess)
            # print(test_fingerprints.shape)  # (batch_size 490)
            # print(test_ground_truth.shape)  # (batch_size, 12)
            validation_fingerprints = fp32_to_uint8(validation_fingerprints)
            output_data = calc(interpreter, validation_fingerprints)
            validation_accuracy = np.mean(np.equal(np.argmax(output_data, axis=1), np.argmax(validation_ground_truth, axis=1)).astype(np.float32))

            batch_size = min(FLAGS.batch_size, set_size - i)
            total_accuracy += (validation_accuracy * batch_size) / set_size

        tf.logging.info('Validation accuracy = %.2f%% (N=%d)' %
                        (total_accuracy * 100, set_size))
        
        # test set
        set_size = audio_processor.set_size('testing')
        tf.logging.info('set_size=%d', set_size)
        total_accuracy = 0
        for i in range(0, set_size, FLAGS.batch_size):
            test_fingerprints, test_ground_truth = audio_processor.get_data(
                FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
            # print(test_fingerprints.shape)  # (batch_size 490)
            # print(test_ground_truth.shape)  # (batch_size, 12)
            test_fingerprints = fp32_to_uint8(test_fingerprints)
            output_data = calc(interpreter, test_fingerprints)
            test_accuracy = np.mean(np.equal(np.argmax(output_data, axis=1), np.argmax(test_ground_truth, axis=1)).astype(np.float32))

            batch_size = min(FLAGS.batch_size, set_size - i)
            total_accuracy += (test_accuracy * batch_size) / set_size

        tf.logging.info('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                                set_size))
    
    elif FLAGS.testing_mode == 'simulate':
        inputs = tf.placeholder(tf.uint8, shape=(None, 49, 10, 1))
        labels = tf.placeholder(tf.float32, shape=(None, 12))
        
        sess2, output_uint8 = simulate_net(inputs)

        predicted_indices = tf.argmax(output_uint8, 1)
        expected_indices = tf.argmax(labels, 1)
        correct_prediction = tf.equal(predicted_indices, expected_indices)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        ########################### simulate lite model ###########################
        
        # training set
        set_size = audio_processor.set_size('training')
        tf.logging.info('set_size=%d', set_size)
        total_accuracy = 0
        for i in range(0, set_size, FLAGS.batch_size):
            training_fingerprints, training_ground_truth = audio_processor.get_data(
                FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'training', sess)
            # print(test_fingerprints.shape)  # (batch_size 490)
            # print(test_ground_truth.shape)  # (batch_size, 12)
            training_fingerprints = fp32_to_uint8(training_fingerprints)

            # # save bin
            # data = training_fingerprints[0]
            # label_index = np.argmax(training_ground_truth, axis=1)[0]
            # save_bin(data, 'test_log/mobilenetv3_quant_gen/bin/data/train/train_{}_{}.bin'.format(i, label_index))

            training_accuracy = sess2.run(evaluation_step, 
                                    feed_dict={inputs: training_fingerprints.reshape(-1, 49, 10, 1), labels: training_ground_truth})

            batch_size = min(FLAGS.batch_size, set_size - i)
            total_accuracy += (training_accuracy * batch_size) / set_size

        tf.logging.info('Training accuracy = %.2f%% (N=%d)' %
                        (total_accuracy * 100, set_size))

        # validation set
        set_size = audio_processor.set_size('validation')
        tf.logging.info('set_size=%d', set_size)
        total_accuracy = 0
        for i in range(0, set_size, FLAGS.batch_size):
            validation_fingerprints, validation_ground_truth = audio_processor.get_data(
                FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'validation', sess)
            # print(test_fingerprints.shape)  # (batch_size 490)
            # print(test_ground_truth.shape)  # (batch_size, 12)
            validation_fingerprints = fp32_to_uint8(validation_fingerprints)

            # # save bin
            # data = validation_fingerprints[0]
            # label_index = np.argmax(validation_ground_truth, axis=1)[0]
            # save_bin(data, 'test_log/mobilenetv3_quant_gen/bin/data/validation/validation_{}_{}.bin'.format(i, label_index))

            validation_accuracy = sess2.run(evaluation_step, 
                                        feed_dict={inputs: validation_fingerprints.reshape(-1, 49, 10, 1), labels: validation_ground_truth})

            batch_size = min(FLAGS.batch_size, set_size - i)
            total_accuracy += (validation_accuracy * batch_size) / set_size

        tf.logging.info('Validation accuracy = %.2f%% (N=%d)' %
                        (total_accuracy * 100, set_size))

        # test set
        set_size = audio_processor.set_size('testing')
        tf.logging.info('set_size=%d', set_size)
        total_accuracy = 0
        for i in range(0, set_size, FLAGS.batch_size):
            test_fingerprints, test_ground_truth = audio_processor.get_data(
                FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
            # print(test_fingerprints.shape)  # (batch_size 490)
            # print(test_ground_truth.shape)  # (batch_size, 12)
            test_fingerprints = fp32_to_uint8(test_fingerprints)

            # # save bin
            # data = test_fingerprints[0]
            # label_index = np.argmax(test_ground_truth, axis=1)[0]
            # save_bin(data, 'test_log/mobilenetv3_quant_gen/bin/data/test/test_{}_{}.bin'.format(i, label_index))

            test_accuracy, output_, labels_ = sess2.run([evaluation_step, output_uint8, labels],
                                feed_dict={inputs: test_fingerprints.reshape(-1, 49, 10, 1), labels: test_ground_truth})

            output_ = output_.astype(np.int32)
            output_ -= 128
            if output_.max() > 127 or output_.min() < -128:
                print("Problem!!!!!!!!", output_.max(), output_.min())
            batch_size = min(FLAGS.batch_size, set_size - i)
            total_accuracy += (test_accuracy * batch_size) / set_size

        tf.logging.info('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                                set_size))

    elif FLAGS.testing_mode == 'debug':
        # Load TFLite model and allocate tensors.
        interpreter2 = tf.lite.Interpreter(model_path='test_log/mobilenetv3_quant_gen/layers_lite_model/stem_conv.lite')
        # interpreter = tf.lite.Interpreter(model_path="tflite_factory/swiftnet-uint8.lite")
        interpreter2.allocate_tensors()

        inputs = tf.placeholder(tf.uint8, shape=(None, 49, 10, 1))
        inputs2 = tf.placeholder(tf.uint8, shape=(None, 25, 5, 16))
        labels = tf.placeholder(tf.float32, shape=(None, 12))
        
        sess3, output_uint8, bfo_soft, aft_soft = debug(inputs, inputs2)

        predicted_indices = tf.argmax(output_uint8, 1)
        expected_indices = tf.argmax(labels, 1)
        correct_prediction = tf.equal(predicted_indices, expected_indices)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        # test set
        set_size = audio_processor.set_size('testing')
        tf.logging.info('set_size=%d', set_size)
        total_accuracy = 0
        for i in range(0, set_size, FLAGS.batch_size):
            test_fingerprints, test_ground_truth = audio_processor.get_data(
                FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
            # print(test_fingerprints.shape)  # (batch_size 490)
            # print(test_ground_truth.shape)  # (batch_size, 12)
            test_fingerprints = fp32_to_uint8(test_fingerprints)
            output_data = calc(interpreter, test_fingerprints)  # 将上层正确的结果送入下层
            output_data2 = calc(interpreter2, test_fingerprints)    # 第二个加法数
            test_accuracy, output_, labels_, bfo_soft_, aft_soft_ = sess3.run([evaluation_step, output_uint8, labels, bfo_soft, aft_soft],
                                feed_dict={inputs: test_fingerprints.reshape(-1, 49, 10, 1), labels: test_ground_truth, inputs2: output_data2})

            batch_size = min(FLAGS.batch_size, set_size - i)
            total_accuracy += (test_accuracy * batch_size) / set_size

            # print(bfo_soft_)
            # print(aft_soft_)
            # print(output_)
            # print(labels_)
            # print()

        tf.logging.info('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                                set_size))

    end_time = time.time()
    tf.logging.info('Running time: {}'.format(end_time - start_time))

    '''
    ############################### get all data mean and std_dev ###############################
    training_fingerprints, training_ground_truth = audio_processor.get_data(
        -1, 0, model_settings, 0.0, 0.0, 0, 'training', sess)
    validation_fingerprints, validation_ground_truth = audio_processor.get_data(
        -1, 0, model_settings, 0.0, 0.0, 0, 'validation', sess)
    testing_fingerprints, testing_ground_truth = audio_processor.get_data(
        -1, 0, model_settings, 0.0, 0.0, 0, 'testing', sess)
    mean_, std_dev = data_stats(training_fingerprints, validation_fingerprints, testing_fingerprints)
    print(mean_, std_dev)
    '''

def main(_):
    # Create the model, load weights from checkpoint and run on train/val/test
    run_inference(FLAGS.wanted_words, FLAGS.sample_rate,
        FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.dct_coefficient_count,
        FLAGS.model_architecture, FLAGS.model_size_info)

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

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)