import tensorflow as tf
import os

from tensorflow.python.tools import freeze_graph


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
	# try:
	# 	converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(
	# 		graph_def_file, input_arrays, output_arrays)
	# except Exception:
	# 	# tf1.14
	# 	converter = tf.lite.TFLiteConverter.from_frozen_graph(
	# 		graph_def_file, input_arrays, output_arrays)
	converter = tf.lite.TFLiteConverter.from_frozen_graph(
		graph_def_file, input_arrays, output_arrays)

	# Official pipeline not working...
	# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
	# converter.optimizations = [tf.lite.Optimize.DEFAULT]	# just quantisize weight to 8bit
	# Set the input and output tensors to uint8 (APIs added in r2.3)
	if post_training_quantize:
		converter.post_training_quantize = True	# True flag is used to convert float model
		converter.inference_type = tf.uint8
		# converter.quantized_input_stats = {input_arrays[0]: (0.0, 1.0)}
		# converter.quantized_input_stats = {input_arrays[0]: (-3.975149608704592, 0.8934739293234528)}
		converter.quantized_input_stats = {input_arrays[0]: (220.81257374779565, 0.8934739293234528)}
		if enable_dummy_quant:
			converter.default_ranges_stats = (0, 6)
	else:
		converter.post_training_quantize = False
		# converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
		# converter.quantized_input_stats = {input_arrays[0]:(-3.975149608704592, 0.8934739293234528)} # mean, std_dev
		# converter.default_ranges_stats = (0, 255)

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


def convert(sess,
            model_dir,
            inference_type="float32",
            output_node_name="logits/activation",
            enable_dummy_quant=False,
            triplet="conv-bn-relu"
            ):
	output_node_names = output_node_name
	inference_type = dtype_dict[inference_type]
	if inference_type == tf.uint8:
		QUANTIZE_FLAG = True
	else:
		QUANTIZE_FLAG = False
	ENABLE_DUMMY_QUANT = enable_dummy_quant
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	tf.logging.info("Freezing Graph...")
	save_to_graphpb_and_freeze(sess,
	                           model_dir=model_dir,
	                           output_node_names=output_node_names)
	tf.logging.info("Freezing complete!")

	tf.logging.info("Converting to TFLite Models...")
	convert_frozen_graph_to_tflite(input_array='fingerprint_input',
	                               output_array=output_node_names,
	                               save_path=model_dir,
	                               post_training_quantize=QUANTIZE_FLAG,
	                               enable_dummy_quant=ENABLE_DUMMY_QUANT)
	# convert_frozen_graph_to_tflite(input_array='fingerprint_input',
	#                                output_array='MBNetV3-CNN/fc/conv/act_quant/FakeQuantWithMinMaxVars',
	#                                save_path=model_dir,
	#                                post_training_quantize=QUANTIZE_FLAG,
	# 							   output_tflite_file='Conv2D.lite',
	#                                enable_dummy_quant=ENABLE_DUMMY_QUANT)
	tf.logging.info("Convert complete!")
	tf.logging.info("Complete!")
	print('Complete!')