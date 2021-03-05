import tensorflow as tf

def apply_activation(tensor, activation='relu'):
	"""
		Applying the activation function to the tensor
	:param tensor: Tensor of any shape.
	:param activation: activation type.
	:return: Tensor after applying the activation function.
	"""
	# Use ELU all the time
	if activation == 'relu':
		tensor = tf.nn.relu(tensor, name='activation')
	elif activation == 'relu6':
		tensor = tf.nn.relu6(tensor, name='activation')
	elif activation == 'elu':
		tensor = tf.nn.elu(tensor, name='activation')
	elif activation == 'leaky':
		tensor = tf.nn.leaky_relu(tensor, alpha=.2, name='activation')
	elif activation == 'swish':
		print("Using swish...")
		tensor = tf.nn.swish(tensor)
	elif activation == 'softmax':
		tf.logging.info("softmax detected!!!!!!")
		tensor = tf.nn.softmax(tensor, name='activation')
	elif activation == 'sigmoid':
		tensor = tf.nn.sigmoid(tensor, name='activation')
	elif activation == 'linear' or activation is None:
		# Uncomment to speedup fp training
		tensor = tensor
		# Uncomment to enable linear bottleneck in quantization
		# tensor = tf.identity(tensor)
	else:
		raise NotImplementedError("Activation %s is not supported." %activation)
	return tensor

def get_activation_fn(activation):
	return lambda tensor: apply_activation(tensor, activation)

def batch_normalization(input,
                        activation='linear',
                        trainable=True,
                        is_training=False,
						momentum=0.96,
						epsilon=1e-3):
	"""
	Batch normalization layer supporting tflite. Fuse activation into batchnorm.
	:param inputs:
	:return:
	"""
	activation_fn = get_activation_fn(activation)
	with tf.variable_scope("bn") as scope:
		output = tf.contrib.layers.batch_norm(input,
		                                      scale=True,
		                                      decay=momentum,
		                                      epsilon=epsilon,
		                                      trainable=trainable,
		                                      reuse=tf.AUTO_REUSE,
		                                      scope=scope,
		                                      activation_fn=activation_fn,
		                                      is_training=is_training,
		                                      fused=True)
	return output


def depthwise_conv2d(inputs,
                     kernel_size,
                     strides=1,
                     padding='SAME',
                     batchnorm=False,
                     depth_multiplier=1,
                     activation='linear',
                     initializer=tf.initializers.glorot_normal(),
                     bias_initializer=tf.constant_initializer(0.00),
                     regularizer=tf.keras.regularizers.l2(5e-5),
                     trainable=True,
                     use_bias=True,
                     is_training=True,
					 name_scope=None
                     ):

	with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE) as scope:
		input_dim = inputs.get_shape()[-1]
		strides_ = (1, strides[0], strides[1], 1)

		output_dim = input_dim * depth_multiplier
		weights = tf.get_variable(shape=[kernel_size[0], kernel_size[1], input_dim, depth_multiplier], name="depthwise_kernel",
								  initializer=initializer, regularizer=regularizer, trainable=trainable)

		activation_fn = get_activation_fn(activation)
		if use_bias:
			bias = tf.get_variable(shape=[output_dim], initializer=bias_initializer, name='bias', trainable=trainable)
			conv_depthwise = tf.nn.bias_add(tf.nn.depthwise_conv2d(input=inputs,
																   filter=weights,
																   strides=strides_,
																   padding=padding), bias, name="depthwise_conv")
		else:
			conv_depthwise = tf.nn.depthwise_conv2d(input=inputs,
													filter=weights,
													strides=strides_,
													padding=padding,
													name="depthwise_conv")
		if batchnorm:
			with tf.variable_scope("depthwise_bn"):
				conv_depthwise = batch_normalization(conv_depthwise,
													 activation=activation,
													 trainable=trainable,
													 is_training=is_training,
													 )
		else:
			conv_depthwise = activation_fn(conv_depthwise)

	return conv_depthwise


def convolution2d(inputs,
                  kernel_size,
                  filters,
                  strides=1,
                  padding='same',
                  batchnorm=False,
                  activation='linear',
				  initializer=tf.initializers.glorot_normal(),
				  bias_initializer=tf.constant_initializer(0.00),
				  regularizer=tf.keras.regularizers.l2(5e-5),
				  trainable=True,
                  use_bias=True,
                  is_training=True,
				  name_scope=None):

	with tf.variable_scope(name_scope, tf.AUTO_REUSE) as scope:
		activation_fn = get_activation_fn(activation)
		conv = tf.layers.conv2d(
			inputs=inputs,
			filters=filters,
			kernel_size=kernel_size,
			strides=strides,
			padding=padding,
			use_bias=use_bias,
			trainable=trainable,
			activation=None,
			kernel_initializer=initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=regularizer,
			name='conv'
		)

		if batchnorm:
			conv = batch_normalization(conv,
									   activation=activation,
									   trainable=trainable,
									   is_training=is_training,
									   )
		else:
			conv = activation_fn(conv)

		return conv

def dense(inputs,
          units,
          activation='relu',
          batchnorm=False,
		  initializer=tf.initializers.glorot_normal(),
		  bias_initializer=tf.constant_initializer(0.00),
		  regularizer=tf.keras.regularizers.l2(5e-5),
		  trainable=True,
          use_bias=True,
          is_training=True,
          name_scope=None):

	with tf.variable_scope(name_scope, tf.AUTO_REUSE) as scope:
		fc = tf.layers.dense(
			inputs=inputs,
			units=units,
			kernel_initializer=initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=regularizer,
			use_bias=use_bias,
			trainable=trainable,
			name=name_scope
		)
		if batchnorm:
			fc = batch_normalization(fc,
			                         activation=activation,
			                         trainable=trainable,
			                         is_training=is_training,
			                         )
		else:
			if activation == "linear" or activation is None:
				# Added for quantization.
				fc = tf.identity(fc, name="activation")
			else:
				activation_fn = get_activation_fn(activation)
				fc = activation_fn(fc)
		return fc
