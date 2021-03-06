#-*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import numpy as np
import time

sys.path.append(os.path.dirname(sys.path[0]))   # 用于上级目录的包调用

from layers import conv2d, depthwise_conv2d, relu, pooling
import input_data_zynq as input_data
import models_zynq as models
# from gen_bin import save_bin

# os.chdir('../')

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
    # scale = (np.round(scale * 2**10) / 2**10).astype(np.float32)
    # add_scale = (np.round(add_scale * 2**10) / 2**10).astype(np.float32)
    scale = np.round(scale * 2**10).astype(np.int32)
    add_scale = np.round(add_scale * 2**10).astype(np.int32)

    # change division to multiplication
    add_scale[2] = np.floor(1 / add_scale[2] * 2**15).astype(np.int32)
    add_scale[5] = np.floor(1 / add_scale[5] * 2**15).astype(np.int32)

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

    if args.save_layers_output == True:
        # define output directory
        layers_output_dir = os.path.join(model_dir, 'layers_output')
        if os.path.exists(layers_output_dir) == False:
            os.mkdir(layers_output_dir)

        # save input
        np.save(os.path.join(layers_output_dir, 'input_data.npy'), input_data)

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
    output = output.astype(np.int32) * s_iwr['stem_conv']
    output = output / 2**10
    
    output = relu(output)
    output += 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    add_2 = output_uint8.copy()   # 给之后的做加法
    # print()

    # save output
    if args.save_layers_output == True:
        np.save(os.path.join(layers_output_dir, 'stem_conv.npy'), output_uint8)

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
    output = output.astype(np.int32) * s_iwr['inverted_residual_1_expansion']
    output = output / 2**10
    
    output = relu(output)
    output += 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    # print()

    # save output
    if args.save_layers_output == True:
        np.save(os.path.join(layers_output_dir, 'inverted_residual_1_expansion.npy'), output_uint8)

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
    output = output.astype(np.int32) * s_iwr['inverted_residual_1_depthwise']
    output = output / 2**10
    
    output = relu(output)
    output += 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    # print()

    # save output
    if args.save_layers_output == True:
        np.save(os.path.join(layers_output_dir, 'inverted_residual_1_depthwise.npy'), output_uint8)

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
    output = output.astype(np.int32) * s_iwr['inverted_residual_1_projection']
    output = output / 2**10 + 128
    
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    add_1 = output_uint8.copy()
    # print()

    # save output
    if args.save_layers_output == True:
        np.save(os.path.join(layers_output_dir, 'inverted_residual_1_projection.npy'), output_uint8)

    ################## inverted residual 1 add ##################
    add_1 = add_1.astype(np.int32)
    add_2 = add_2.astype(np.int32)

    # add_1 = tf.constant(0.1732778549194336, tf.float32) * (add_1 - 128)
    # add_2 = tf.constant(0.20100615918636322, tf.float32) * (add_2 - 128)
    add_1 = s_add['inverted_residual_1_add'][0] * (add_1 - 128)
    add_2 = s_add['inverted_residual_1_add'][1] * (add_2 - 128)

    output_result = add_1 + add_2
    # output = output_result / tf.constant(0.26455792784690857, tf.float32) + 128
    output = output_result * s_add['inverted_residual_1_add'][2]
    output = output / 2**15 + 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)

    # save output
    if args.save_layers_output == True:
        np.save(os.path.join(layers_output_dir, 'inverted_residual_1_add.npy'), output_uint8)

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
    output = output.astype(np.int32) * s_iwr['inverted_residual_2_expansion']
    output = output / 2**10

    output = relu(output)
    output += 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    # print()

    # save output
    if args.save_layers_output == True:
        np.save(os.path.join(layers_output_dir, 'inverted_residual_2_expansion.npy'), output_uint8)

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
    output = output.astype(np.int32) * s_iwr['inverted_residual_2_depthwise']
    output = output / 2**10
    
    output = relu(output)
    output += 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    # print()

    # save output
    if args.save_layers_output == True:
        np.save(os.path.join(layers_output_dir, 'inverted_residual_2_depthwise.npy'), output_uint8)

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
    output = output.astype(np.int32) * s_iwr['inverted_residual_2_projection']
    output = output / 2**10 + 128
    
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    add_2 = output_uint8.copy()
    # print()

    # save output
    if args.save_layers_output == True:
        np.save(os.path.join(layers_output_dir, 'inverted_residual_2_projection.npy'), output_uint8)

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
    output = output.astype(np.int32) * s_iwr['inverted_residual_3_expansion']
    output = output / 2**10
    
    output = relu(output)
    output += 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    # print()

    # save output
    if args.save_layers_output == True:
        np.save(os.path.join(layers_output_dir, 'inverted_residual_3_expansion.npy'), output_uint8)

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
    output = output.astype(np.int32) * s_iwr['inverted_residual_3_depthwise']
    output = output / 2**10
    
    output = relu(output)
    output += 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    # print()

    # save output
    if args.save_layers_output == True:
        np.save(os.path.join(layers_output_dir, 'inverted_residual_3_depthwise.npy'), output_uint8)

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
    output = output.astype(np.int32) * s_iwr['inverted_residual_3_projection']
    output = output / 2**10 + 128
    
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    add_1 = output_uint8.copy()
    # print()

    # save output
    if args.save_layers_output == True:
        np.save(os.path.join(layers_output_dir, 'inverted_residual_3_projection.npy'), output_uint8)

    ################## inverted residual 3 add ##################
    add_1 = add_1.astype(np.int32)
    add_2 = add_2.astype(np.int32)

    # add_1 = tf.constant(0.19232141971588135, tf.float32) * (add_1 - 128)
    # add_2 = tf.constant(0.12740808725357056, tf.float32) * (add_2 - 128)
    add_1 = s_add['inverted_residual_3_add'][0] * (add_1 - 128)
    add_2 = s_add['inverted_residual_3_add'][1] * (add_2 - 128)

    output_result = add_1 + add_2
    # output_uint8 = output_result / tf.constant(0.20970593392848969, tf.float32) + 128
    output = output_result * s_add['inverted_residual_3_add'][2]
    output = output / 2**15 + 128
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)

    # save output
    if args.save_layers_output == True:
        np.save(os.path.join(layers_output_dir, 'inverted_residual_3_add.npy'), output_uint8)

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

    # save output
    if args.save_layers_output == True:
        np.save(os.path.join(layers_output_dir, 'AvgPool.npy'), output_uint8)

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
    output = output.astype(np.int32) * s_iwr['Conv2D']
    output = output / 2**10 + 128
    
    output_uint8 = output.round()
    output_uint8 = np.clip(output_uint8, 0, 255).astype(np.uint8)
    # print()

    # save output
    if args.save_layers_output == True:
        np.save(os.path.join(layers_output_dir, 'Conv2D.npy'), output_uint8)

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

def run_inference(args, wanted_words, sample_rate, clip_duration_ms,
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

    
    words_list = input_data.prepare_words_list(wanted_words.split(','))
    model_settings = models.prepare_model_settings(
        len(words_list), sample_rate, clip_duration_ms, window_size_ms,
        window_stride_ms, dct_coefficient_count)

    audio_processor = input_data.AudioProcessor(
        args.data_url, args.data_dir, args.silence_percentage,
        args.unknown_percentage,
        args.wanted_words.split(','), args.validation_percentage,
        args.testing_percentage, model_settings)
    

    start_time = time.time()

    ########################### simulate lite model ###########################
    '''
    # training set
    set_size = audio_processor.set_size('training')
    print('set_size=%d', set_size)
    total_accuracy = 0
    for i in range(0, set_size, args.batch_size):
        training_fingerprints, training_ground_truth = audio_processor.get_data(
            args.batch_size, i, model_settings, 0.0, 0.0, 0, 'training')
        # print(test_fingerprints.shape)  # (batch_size 490)
        # print(test_ground_truth.shape)  # (batch_size, 12)
        training_fingerprints = fp32_to_uint8(training_fingerprints)

        # # save bin
        # data = training_fingerprints[0]
        # label_index = np.argmax(training_ground_truth, axis=1)[0]
        # save_bin(data, 'test_log/mobilenetv3_quant_gen/bin/data/train/train_{}_{}.bin'.format(i, label_index))

        # calculate accuracy
        output_uint8 = simulate_net(training_fingerprints.reshape(-1, 49, 10, 1))
        predicted_indices = np.argmax(output_uint8, 1)
        expected_indices = np.argmax(training_ground_truth, 1)
        correct_prediction = np.equal(predicted_indices, expected_indices)
        training_accuracy = np.mean(correct_prediction.astype(np.float32))

        batch_size = min(args.batch_size, set_size - i)
        total_accuracy += (training_accuracy * batch_size) / set_size

    print('Training accuracy = %.2f%% (N=%d)' %
                    (total_accuracy * 100, set_size))

    # validation set
    set_size = audio_processor.set_size('validation')
    print('set_size=%d', set_size)
    total_accuracy = 0
    for i in range(0, set_size, args.batch_size):
        validation_fingerprints, validation_ground_truth = audio_processor.get_data(
            args.batch_size, i, model_settings, 0.0, 0.0, 0, 'validation')
        # print(test_fingerprints.shape)  # (batch_size 490)
        # print(test_ground_truth.shape)  # (batch_size, 12)
        validation_fingerprints = fp32_to_uint8(validation_fingerprints)

        # # save bin
        # data = validation_fingerprints[0]
        # label_index = np.argmax(validation_ground_truth, axis=1)[0]
        # save_bin(data, 'test_log/mobilenetv3_quant_gen/bin/data/validation/validation_{}_{}.bin'.format(i, label_index))

        # calculate accuracy
        output_uint8 = simulate_net(validation_fingerprints.reshape(-1, 49, 10, 1))
        predicted_indices = np.argmax(output_uint8, 1)
        expected_indices = np.argmax(validation_ground_truth, 1)
        correct_prediction = np.equal(predicted_indices, expected_indices)
        validation_accuracy = np.mean(correct_prediction.astype(np.float32))

        batch_size = min(args.batch_size, set_size - i)
        total_accuracy += (validation_accuracy * batch_size) / set_size

    print('Validation accuracy = %.2f%% (N=%d)' %
                    (total_accuracy * 100, set_size))
    '''

    # test set
    set_size = audio_processor.set_size('testing')
    print('set_size=%d', set_size)
    total_accuracy = 0
    for i in range(0, set_size, args.batch_size):
        test_fingerprints, test_ground_truth = audio_processor.get_data(
            args.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing')
        # print(test_fingerprints.shape)  # (batch_size 490)
        # print(test_ground_truth.shape)  # (batch_size, 12)
        test_fingerprints = fp32_to_uint8(test_fingerprints)

        # # save bin
        # data = test_fingerprints[0]
        # label_index = np.argmax(test_ground_truth, axis=1)[0]
        # save_bin(data, 'test_log/mobilenetv3_quant_gen/bin/data/test/test_{}_{}.bin'.format(i, label_index))

        # calculate accuracy
        output_uint8 = simulate_net(test_fingerprints.reshape(-1, 49, 10, 1))
        predicted_indices = np.argmax(output_uint8, 1)
        expected_indices = np.argmax(test_ground_truth, 1)
        correct_prediction = np.equal(predicted_indices, expected_indices)
        test_accuracy = np.mean(correct_prediction.astype(np.float32))
        
        # save correct inputs
        # if test_accuracy == 1.:
        #     # save_bin(test_fingerprints[0], 'test_log/mobilenetv3_quant_mfcc_gen/bin/{}.bin'.format(words_list[expected_indices[0]]))
        #     np.save('test_log/mobilenetv3_quant_mfcc_gen/input_data/{}_{}.npy'.format(i, words_list[expected_indices[0]]), test_fingerprints[0])
        #     # sys.exit(0)

        # save all inputs
        # np.save('test_log/mobilenetv3_quant_mfcc_gen/input_data/{}_{}.npy'.format(i, words_list[expected_indices[0]]), test_fingerprints[0])

        # save intermediate data
        if args.save_layers_output == True:
            print('Save complete')
            sys.exit(0)

        batch_size = min(args.batch_size, set_size - i)
        total_accuracy += (test_accuracy * batch_size) / set_size

    print('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                            set_size))

    end_time = time.time()
    print('Running time: {} second'.format(end_time - start_time))


    '''
    ############################### get all data mean and std_dev ###############################
    training_fingerprints, training_ground_truth = audio_processor.get_data(
        -1, 0, model_settings, 0.0, 0.0, 0, 'training')
    validation_fingerprints, validation_ground_truth = audio_processor.get_data(
        -1, 0, model_settings, 0.0, 0.0, 0, 'validation')
    testing_fingerprints, testing_ground_truth = audio_processor.get_data(
        -1, 0, model_settings, 0.0, 0.0, 0, 'testing')
    mean_, std_dev = data_stats(training_fingerprints, validation_fingerprints, testing_fingerprints)
    print(mean_, std_dev)
    '''

def main(args):
    # Create the model, load weights from checkpoint and run on train/val/test
    run_inference(args, args.wanted_words, args.sample_rate,
        args.clip_duration_ms, args.window_size_ms,
        args.window_stride_ms, args.dct_coefficient_count,
        args.model_architecture, args.model_size_info)

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
    parser.add_argument(
        '--save_layers_output',
        action='store_true',
        default=False,
        help='Save the output of each layers'
    )

    args = parser.parse_args()
    main(args)
