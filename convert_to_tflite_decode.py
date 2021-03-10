import tensorflow as tf
import os

from tensorflow.python.tools import freeze_graph
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.framework import graph_util

import input_data
import models

def convert_frozen_graph_to_tflite(save_path=None,
                                   graph_name='frozen_graph.pb',
                                   input_array='fingerprint_input',
                                   output_array='logits/SoftMax',
                                   output_tflite_file="swiftnet-uint8.lite",
                                   post_training_quantize=True,
                                   enable_dummy_quant=True):
	"""
	Convert the frozen graph pb to tflite model (8-bit)
	:param graph_name: name of frozen graph
	:param input_array: input nodes array
	:param output_array: output nodes array
	:param output_tflite_file: Output tflite file name.
	:param post_training_quantize: Whether to enable post training quantization
	:param
	:return:
	"""
	if not isinstance(input_array, list):
		input_arrays = [input_array]
	else:
		input_arrays = input_array
	if not isinstance(output_array, list):
		output_arrays = [output_array]
	else:
		output_arrays = output_array

	graph_def_file = os.path.join(save_path, graph_name)
	try:
		converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
			graph_def_file, input_arrays, output_arrays)
	except Exception:
		# tf1.14
		converter = tf.lite.TFLiteConverter.from_frozen_graph(
			graph_def_file, input_arrays, output_arrays)

	# Official pipeline not working...
	# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
	converter.optimizations = [tf.lite.Optimize.DEFAULT]	# just quantisize weight to 8bit
	# if post_training_quantize:
	# 	# converter.post_training_quantize = True	# True flag is used to convert float model
	# 	converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
	# 	converter.quantized_input_stats = {input_arrays[0]: (0.0, 1.0)}
	# 	if enable_dummy_quant:
	# 		converter.default_ranges_stats = (0, 6)
	# else:
	# 	converter.post_training_quantize = False

	tflite_model = converter.convert()
	tflite_output_path = os.path.join(save_path, output_tflite_file)
	open(tflite_output_path, "wb").write(tflite_model)


def save_to_graphpb_and_freeze(sess,
                               output_node_names,
                               model_dir,
                               graph_name="frozen_graph.pb"):
	"""
	:param pb_log_dir:
	:param output_node_names:
	:param graph_name:
	:return:
	"""

	pb_txt_name = graph_name + ".pbtxt"
	pb_txt_path = os.path.join(model_dir, pb_txt_name)
	pb_output_path = os.path.join(model_dir, graph_name)

	tf.train.write_graph(sess.graph_def, model_dir, pb_txt_name, as_text=True)

	ckpt_dir = model_dir
	ckpt_path = tf.train.latest_checkpoint(ckpt_dir)

	# Construct a copy for debug
	saver = tf.train.Saver()
	save_path_ = os.path.join(model_dir, "model-lite.ckpt")
	saver.save(sess, save_path_)

	freeze_graph.freeze_graph(input_graph=pb_txt_path, input_saver='', input_binary=False,
	                          input_checkpoint=ckpt_path, output_node_names=output_node_names,
	                          restore_op_name='save/restore_all', filename_tensor_name='save/Const:0',
	                          output_graph=pb_output_path, clear_devices=True, initializer_nodes='')

dtype_dict = {
    "float32": tf.float32,
    "uint8": tf.uint8
}

def create_inference_graph(wanted_words, sample_rate, clip_duration_ms,
                           clip_stride_ms, window_size_ms, window_stride_ms,
                           dct_coefficient_count, model_architecture, model_size_info, output_node_name):
	"""Creates an audio model with the nodes needed for inference.

	Uses the supplied arguments to create a model, and inserts the input and
	output nodes that are needed to use the graph for inference.

	Args:
		wanted_words: Comma-separated list of the words we're trying to recognize.
		sample_rate: How many samples per second are in the input audio files.
		clip_duration_ms: How many samples to analyze for the audio pattern.
		clip_stride_ms: How often to run recognition. Useful for models with cache.
		window_size_ms: Time slice duration to estimate frequencies from.
		window_stride_ms: How far apart time slices should be.
		dct_coefficient_count: Number of frequency bands to analyze.
		model_architecture: Name of the kind of model to generate.
	"""

	words_list = input_data.prepare_words_list(wanted_words.split(','))
	model_settings = models.prepare_model_settings(
		len(words_list), sample_rate, clip_duration_ms, window_size_ms,
		window_stride_ms, dct_coefficient_count)
	runtime_settings = {'clip_stride_ms': clip_stride_ms}

	wav_data_placeholder = tf.placeholder(tf.string, [], name='wav_data')
	decoded_sample_data = contrib_audio.decode_wav(
		wav_data_placeholder,
		desired_channels=1,
		desired_samples=model_settings['desired_samples'],
		name='decoded_sample_data')
	spectrogram = contrib_audio.audio_spectrogram(
		decoded_sample_data.audio,
		window_size=model_settings['window_size_samples'],
		stride=model_settings['window_stride_samples'],
		magnitude_squared=True)
	fingerprint_input = contrib_audio.mfcc(
		spectrogram,
		decoded_sample_data.sample_rate,
		dct_coefficient_count=dct_coefficient_count)
	fingerprint_frequency_size = model_settings['dct_coefficient_count']
	fingerprint_time_size = model_settings['spectrogram_length']
	reshaped_input = tf.reshape(fingerprint_input, [
		-1, fingerprint_time_size * fingerprint_frequency_size
	])

	logits = models.create_model(
		reshaped_input, model_settings, model_architecture, model_size_info,
		is_training=False, runtime_settings=runtime_settings)

	# Create an output to use for inference.
	tf.nn.softmax(logits, name=output_node_name)

def freeze(sess, FLAGS, model_dir, file_name='frozen_graph.pb'):
	# Turn all the variables into inline constants inside the graph and save it.
	frozen_graph_def = graph_util.convert_variables_to_constants(
		sess, sess.graph_def, ['labels_softmax'])
	tf.train.write_graph(
		frozen_graph_def,
		os.path.dirname(model_dir),
		os.path.basename(file_name),
		as_text=False)
	tf.logging.info('Saved frozen graph to %s', file_name)

def convert(sess,
            model_dir,
			FLAGS,
            inference_type="float32",
            output_node_name="logits/activation",
            enable_dummy_quant=False,
            triplet="conv-bn-relu",
            ):
	
	create_inference_graph(FLAGS.wanted_words, FLAGS.sample_rate,
                         FLAGS.clip_duration_ms, FLAGS.clip_stride_ms,
                         FLAGS.window_size_ms, FLAGS.window_stride_ms,
                         FLAGS.dct_coefficient_count, FLAGS.model_architecture,
                         FLAGS.model_size_info, output_node_name)
	
	# output_node_names = output_node_name
	inference_type = dtype_dict[inference_type]
	if inference_type == tf.uint8:
		QUANTIZE_FLAG = True
	else:
		QUANTIZE_FLAG = False

	ENABLE_DUMMY_QUANT = enable_dummy_quant
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	tf.logging.info("Freezing Graph...")
	# save_to_graphpb_and_freeze(sess,
	#                            model_dir=model_dir,
	#                            output_node_names=output_node_name)
	freeze(sess, FLAGS, model_dir)
	tf.logging.info("Freezing complete!")

	# tf.logging.info("Converting to TFLite Models...")
	# convert_frozen_graph_to_tflite(input_array='fingerprint_input',
	#                                output_array=output_node_name,
	#                                save_path=model_dir,
	#                                post_training_quantize=QUANTIZE_FLAG,
	#                                enable_dummy_quant=ENABLE_DUMMY_QUANT)
	# tf.logging.info("Convert complete!")
	# tf.logging.info("Complete!")
