#-*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import numpy as np
import tensorflow as tf

import input_data
import models


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

    # method 2
    std_dev = 0.8934739293234528
    mean_ = 220.81257374779565
    q = r / std_dev + mean_
    q = q.astype(np.uint8)
    return q

def simulate_net(input_data, label):
    sess = tf.Session()
    ################## stem conv ##################
    # print('stem conv')
    # new_data = input_data.reshape(-1, 49, 10, 1)
    # new_data = tf.reshape(input_data, [-1, 49, 10, 1])
    # new_data = tf.convert_to_tensor(new_data, tf.float32, name='input')
    new_data = tf.cast(input_data, tf.float32)
    new_data =  new_data - 221.
    s_iw = 1.1192268133163452 * 0.0009845112217590213

    weight = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_stem_conv_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_stem_conv_conv_Conv2D_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32, name='weight')
    weight = weight - 132.
    weight = tf.transpose(weight, perm=[1,2,0,3])   # [filter_height, filter_width, in_channels, out_channels]

    bias = tf.convert_to_tensor(bias, tf.float32, name='bias')
    bias = 0.0011018913937732577 * bias
    # print(weight)
    # print(bias)

    output = tf.nn.depthwise_conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,2,2,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None,  # 数据格式，与步长参数配合，决定移动方式
                name='stem_conv') # 名字，用于tensorboard图形显示时使用
    output_fp = s_iw * output

    output_add = tf.add(output_fp, bias, name='add')
    # output_uint8 = tf.cast(output, dtype=tf.uint8)
    output_add = output_add / 0.16148914396762848
    
    output_relu = tf.nn.relu(output_add)
    output_uint8 = tf.math.round(output_relu, name='round')
    output_uint8 = tf.cast(output_uint8, tf.uint8, name='uint8')
    add_2 = tf.identity(output_uint8)   # 给之后的做加法
    # print()

    ################## inverted residual 1 expansion ##################
    # print('inverted residual 1 expansion')
    new_data = tf.cast(output_uint8, tf.float32)
    s_iw = 0.16148914396762848 * 0.01888326182961464

    weight = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_1_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_1_expansion_conv_Conv2D_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 146
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    bias *= 0.003049441846087575
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = s_iw * output
    output = output + bias
    output = output / 0.27361148595809937
    
    output = tf.nn.relu(output)
    output_uint8 = tf.math.round(output)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()

    ################## inverted residual 1 depthwise ##################
    # print('inverted residual 1 depthwise')
    new_data = tf.cast(output_uint8, tf.float32)
    s_iw = 0.27361148595809937 * 0.016701024025678635

    weight = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_1_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_1_depthwise_depthwise_conv_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 127
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    bias *= 0.0045695919543504715
    # print(bias)

    output = tf.nn.depthwise_conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = s_iw * output
    output = output + bias
    output = output / 0.12676289677619934
    
    output = tf.nn.relu(output)
    output_uint8 = tf.math.round(output)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()

    ################## inverted residual 1 projection ##################
    # print('inverted residual 1 projection')
    new_data = tf.cast(output_uint8, tf.float32)
    s_iw = 0.12676289677619934 * 0.007413254585117102

    weight = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_1_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_1_projection_conv_Conv2D_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 101
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    bias *= 0.0009397256653755903
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = s_iw * output
    output = output + bias
    output = output / 0.16901935636997223 + 133
    
    output = tf.nn.relu(output)
    output_uint8 = tf.math.round(output)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    add_1 = tf.identity(output_uint8)
    # print()

    ################## inverted residual 1 add ##################
    add_1 = tf.cast(add_1, tf.float32)
    add_2 = tf.cast(add_2, tf.float32)

    add_1 = 0.16901935636997223 * (add_1 - 133)
    add_2 = 0.16148914396762848 * add_2

    output_result = tf.add(add_1, add_2)
    output_uint8 = output_result / 0.24699252843856812 + 89
    output_uint8 = tf.math.round(output_uint8)
    output_uint8 = tf.cast(output_uint8, tf.uint8)

    ################## inverted residual 2 expansion ##################
    # print('inverted residual 2 expansion')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 89
    s_iw = 0.24699252843856812 * 0.008363455533981323

    weight = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_2_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_2_expansion_conv_Conv2D_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 149
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    bias *= 0.0020657109562307596
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = s_iw * output
    output = output + bias
    output = output / 0.09814818948507309
    
    output = tf.nn.relu(output)
    output_uint8 = tf.math.round(output)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()

    ################## inverted residual 2 depthwise ##################
    # print('inverted residual 2 depthwise')
    new_data = tf.cast(output_uint8, tf.float32)
    s_iw = 0.09814818948507309 * 0.014716151170432568

    weight = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_2_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_2_depthwise_depthwise_conv_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 120
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    bias *= 0.0014443636173382401
    # print(bias)

    output = tf.nn.depthwise_conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = s_iw * output
    output = output + bias
    output = output / 0.062810979783535
    
    output = tf.nn.relu(output)
    output_uint8 = tf.math.round(output)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()

    ################## inverted residual 2 projection ##################
    # print('inverted residual 2 projection')
    new_data = tf.cast(output_uint8, tf.float32)
    s_iw = 0.062810979783535 * 0.006514572538435459

    weight = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_2_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_2_projection_conv_Conv2D_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 148
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    bias *= 0.00040918667218647897
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = s_iw * output
    output = output + bias
    output = output / 0.0929793044924736 + 138
    
    output = tf.nn.relu(output)
    output_uint8 = tf.math.round(output)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    add_2 = tf.identity(output_uint8)
    # print()

    ################## inverted residual 3 expansion ##################
    # print('inverted residual 3 expansion')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 138
    s_iw = 0.0929793044924736 * 0.005988169927150011

    weight = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_3_expansion_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_3_expansion_conv_Conv2D_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 137
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    bias *= 0.0005567758926190436
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = s_iw * output
    output = output + bias
    output = output / 0.07842949777841568
    
    output = tf.nn.relu(output)
    output_uint8 = tf.math.round(output)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()

    ################## inverted residual 3 depthwise ##################
    # print('inverted residual 3 depthwise')
    new_data = tf.cast(output_uint8, tf.float32)
    s_iw = 0.07842949777841568 * 0.17394107580184937

    weight = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_3_depthwise_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_3_depthwise_depthwise_conv_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 79
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    bias *= 0.013642110861837864
    # print(bias)

    output = tf.nn.depthwise_conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = s_iw * output
    output = output + bias
    output = output / 0.05131378769874573
    
    output = tf.nn.relu(output)
    output_uint8 = tf.math.round(output)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    # print()

    ################## inverted residual 3 projection ##################
    # print('inverted residual 3 projection')
    new_data = tf.cast(output_uint8, tf.float32)
    s_iw = 0.05131378769874573 * 0.01676042005419731

    weight = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_3_projection_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_inverted_residual_3_projection_conv_Conv2D_Fold_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 125
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    bias *= 0.0008600406581535935
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = s_iw * output
    output = output + bias
    output = output / 0.20826007425785065 + 133
    
    output = tf.nn.relu(output)
    output_uint8 = tf.math.round(output)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    add_1 = tf.identity(output_uint8)
    # print()

    ################## inverted residual 1 add ##################
    add_1 = tf.cast(add_1, tf.float32)
    add_2 = tf.cast(add_2, tf.float32)

    add_1 = 0.20826007425785065 * (add_1 - 133)
    add_2 = 0.0929793044924736 * (add_2 - 138)

    output_result = tf.add(add_1, add_2)
    output_uint8 = output_result / 0.21021947264671326 + 131
    output_uint8 = tf.math.round(output_uint8)
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
    output_uint8 = tf.math.round(output)
    output_uint8 = tf.cast(output, tf.uint8)

    ################## Conv2D ##################
    # print('Conv2D')
    new_data = tf.cast(output_uint8, tf.float32)
    new_data -= 131
    s_iw = 0.21021947264671326 * 0.01610618270933628

    weight = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_fc_conv_weights_quant_FakeQuantWithMinMaxVars.npy')
    bias = np.load('test_log/mobilenetv3_quant_eval/weight/MBNetV3-CNN_fc_conv_Conv2D_bias.npy')
    # print(weight.dtype, weight.shape)
    # print(bias.dtype, bias.shape)

    weight = tf.convert_to_tensor(weight, tf.float32)
    weight -= 143
    weight = tf.transpose(weight, perm=[1,2,3,0])
    # print(weight)

    bias = tf.convert_to_tensor(bias, tf.float32)
    bias *= 0.0033858332317322493
    # print(bias)

    output = tf.nn.conv2d(new_data,  # 张量输入
                filter=weight, # 卷积核参数
                strides=[1,1,1,1], # 步长参数
                padding="SAME", # 卷积方式
                data_format=None)  # 数据格式，与步长参数配合，决定移动方式
    output = s_iw * output
    output = output + bias
    output = output / 0.1784215271472931 + 129
    
    output = tf.nn.relu(output)
    output_uint8 = tf.math.round(output)
    output_uint8 = tf.cast(output_uint8, tf.uint8)
    add_1 = tf.identity(output_uint8)
    # print()

    ################## Reshape ##################
    output_uint8 = tf.squeeze(output_uint8, axis=[1,2])

    ################## Softmax ##################
    new_data = tf.cast(output_uint8, tf.float32)
    new_data = 0.1784215271472931 * (new_data - 129)
    output = tf.nn.softmax(new_data)
    output = output / 0.00390625

    output_uint8 = tf.math.round(output)
    output_uint8 = tf.cast(output_uint8, tf.uint8)

    ################## running ##################
    # return np.mean(np.equal(np.argmax(output_uint8, axis=1), np.argmax(label, axis=1)).astype(np.float32))
    return (sess, output_uint8)

def calc(interpreter, input_data, label):

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

    return np.mean(np.equal(np.argmax(output_data, axis=1), np.argmax(label, axis=1)).astype(np.float32))

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
            training_accuracy = calc(interpreter, training_fingerprints, training_ground_truth)

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
            validation_accuracy = calc(interpreter, validation_fingerprints, validation_ground_truth)

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
            test_accuracy = calc(interpreter, test_fingerprints, test_ground_truth)

            batch_size = min(FLAGS.batch_size, set_size - i)
            total_accuracy += (test_accuracy * batch_size) / set_size

        tf.logging.info('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                                set_size))
    
    elif FLAGS.testing_mode == 'simulate':
        inputs = tf.placeholder(tf.uint8, shape=(None, 49, 10, 1))
        labels = tf.placeholder(tf.float32, shape=(None, 12))
        
        sess2, output_uint8 = simulate_net(inputs, labels)

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
            test_accuracy = sess2.run(evaluation_step, 
                                feed_dict={inputs: test_fingerprints.reshape(-1, 49, 10, 1), labels: test_ground_truth})

            batch_size = min(FLAGS.batch_size, set_size - i)
            total_accuracy += (test_accuracy * batch_size) / set_size

        tf.logging.info('Test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
                                                                set_size))

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