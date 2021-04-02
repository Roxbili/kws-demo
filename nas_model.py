from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os

from utilities import depthwise_conv2d, convolution2d, dense

def _add_to_name_scope(name, scope):
	return os.path.join(scope, name)


def inverted_resdidual_block(inputs,
							 num_outputs,
							 kernel_size,
							 stride,
							 expansion_factor,
                             is_training=True):
	num_inputs = int(inputs.get_shape()[-1])
	expansion_filters = num_inputs * expansion_factor


	expansion = convolution2d(inputs,
							  filters=expansion_filters,
							  kernel_size=1,
							  strides=1,
							  name_scope="expansion",
							  batchnorm=True,
							  use_bias=False,
							  activation="relu",
	                          is_training=is_training)

	depthwise = depthwise_conv2d(expansion,
								 depth_multiplier=1,
								 strides=stride,
								 kernel_size=kernel_size,
								 name_scope="depthwise",
								 batchnorm=True,
								 use_bias=False,
								 activation='relu',
	                             is_training=is_training
								 )

	projection = convolution2d(depthwise,
							   filters=num_outputs,
							   kernel_size=1,
							   strides=1,
							   batchnorm=True,
							   name_scope="projection",
	                           use_bias=False,
	                           activation='linear',
	                           is_training=is_training)

	if stride[0] == 1 and stride[1] == 1 and num_inputs == num_outputs:
		return tf.identity(tf.add(projection, inputs, name="residual_add"))
	else:
		return tf.identity(projection, name="residual")	# 返回和input一样的tensor



def create_mbnetv3_cnn_model(fingerprint_input, model_settings,
							 is_training):
	"""Builds a model with depthwise separable convolutional neural network
	Model definition is based on https://arxiv.org/abs/1704.04861 and
	Tensorflow implementation: https://github.com/Zehaos/MobileNet

	model_size_info: defines number of layers, followed by the DS-Conv layer
	  parameters in the order {number of conv features, conv filter height,
	  width and stride in y,x dir.} for each of the layers.
	Note that first layer is always regular convolution, but the remaining
	  layers are all depthwise separable convolutions.
	"""


	label_count = model_settings['label_count']
	input_frequency_size = model_settings['dct_coefficient_count']
	input_time_size = model_settings['spectrogram_length']
	fingerprint_4d = tf.reshape(fingerprint_input,
								[-1, input_time_size, input_frequency_size, 1])

	t_dim = input_time_size
	f_dim = input_frequency_size


	model_size_info = [
		4,
		16, 10, 4, 2, 2,
		16, 3, 3, 1, 1, 2,
		32, 3, 3, 1, 1, 2,
		32, 5, 5, 1, 1, 2,
	]
	# Extract model dimensions from model_size_info
	num_layers = model_size_info[0]

	print("Total number of layers: %d")
	conv_feat = [None] * num_layers
	conv_kt = [None] * num_layers
	conv_kf = [None] * num_layers
	conv_st = [None] * num_layers
	conv_sf = [None] * num_layers
	conv_expansion_factor = [None] * num_layers
	i = 1
	for layer_no in range(0, num_layers):
		conv_feat[layer_no] = model_size_info[i]
		i += 1
		conv_kt[layer_no] = model_size_info[i]
		i += 1
		conv_kf[layer_no] = model_size_info[i]
		i += 1
		conv_st[layer_no] = model_size_info[i]
		i += 1
		conv_sf[layer_no] = model_size_info[i]
		i += 1
		if layer_no != 0:
			conv_expansion_factor[layer_no] = model_size_info[i]
			i += 1

	scope = 'MBNetV3-CNN'

	with tf.variable_scope(scope) as sc:
		for layer_no in range(0, num_layers):
			if layer_no == 0:
				net = convolution2d(fingerprint_4d,
									filters=conv_feat[layer_no],
									kernel_size=(conv_kt[layer_no], conv_kf[layer_no]),
									strides=(conv_st[layer_no], conv_sf[layer_no]),
									batchnorm=True,
									activation="relu",
									padding="SAME",
				                    name_scope="stem_conv",
				                    use_bias=False,
				                    is_training=is_training)
			else:
				with tf.variable_scope("inverted_residual_%d" %layer_no, tf.AUTO_REUSE) as scope:
					net = inverted_resdidual_block(net,
												   num_outputs=conv_feat[layer_no],
												   kernel_size=[conv_kt[layer_no], conv_kf[layer_no]],
												   expansion_factor=conv_expansion_factor[layer_no],
												   stride=[1, 1],
					                               is_training=is_training
					                               )
			t_dim = math.ceil(t_dim / float(conv_st[layer_no]))	# math.ceil 向上取整
			f_dim = math.ceil(f_dim / float(conv_sf[layer_no]))

		# t_dim = 25, f_dim = 5
		net = tf.layers.average_pooling2d(net, pool_size=[t_dim, f_dim],
		                                  strides=[t_dim, f_dim], name='pool')

		logits = convolution2d(net, kernel_size=1, filters=label_count,	# label_count = 12
		                       activation=None, batchnorm=None,
		                       use_bias=True, is_training=is_training, name_scope="fc")
		logits = tf.layers.flatten(logits)

	return logits