#-*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os, sys
import numpy as np
import time

# sys.path.append('../')

from layers import conv2d, depthwise_conv2d, relu, pooling

# os.chdir('../')

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

def run_inference():
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

    
    words_list = 'silence,unknown,yes,no,up,down,left,right,on,off,stop,go'.split(',')

    start_time = time.time()

    ########################### simulate lite model ###########################

    # test set
    input_data_dir = os.path.join(model_dir, 'input_data')
    data_path = os.listdir(input_data_dir)
    data_path.sort()
    set_size = 4890.
    total_accuracy = 0.
    for i in range(len(data_path)):
        path = os.path.join(input_data_dir, data_path[i])
        test_fingerprints = np.load(path)

        # # save bin
        # data = test_fingerprints[0]
        # label_index = np.argmax(test_ground_truth, axis=1)[0]
        # save_bin(data, 'test_log/mobilenetv3_quant_gen/bin/data/test/test_{}_{}.bin'.format(i, label_index))

        # calculate accuracy
        output_uint8 = simulate_net(test_fingerprints.reshape(-1, 49, 10, 1))
        predicted_indices = np.argmax(output_uint8, 1)
        test_accuracy = 0.
        if words_list[predicted_indices[0]] in path:
            test_accuracy = 1.
        
        batch_size = 1
        total_accuracy += (test_accuracy * batch_size) / set_size

    print('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                            set_size))

    end_time = time.time()
    print('Running time: {} second'.format(end_time - start_time))

def main():
    # Create the model, load weights from checkpoint and run on train/val/test
    run_inference()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_layers_output',
        action='store_true',
        default=False,
        help='Save the output of each layers'
    )
    args = parser.parse_args()

    # model_dir = 'test_log/mobilenetv3_quant_gen'
    model_dir = 'test_log/mobilenetv3_quant_mfcc_gen'

    main()
