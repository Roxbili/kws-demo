import tensorflow as tf
import os, sys
import argparse

from tensorflow.python.tools import freeze_graph
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.framework import graph_util

import input_data
import models
from convert_to_tflite import convert as convert2

def convert_frozen_graph_to_tflite(save_path=None,
                                   graph_name='frozen_graph.pb',
                                   input_array='fingerprint_input',
                                   output_array='logits/SoftMax',
                                   output_tflite_file="swiftnet-uint8.lite",
                                   quantisize_type=None,
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
    if quantisize_type == 'weight':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]    # just quantisize weight to 8bit
    elif quantisize_type == 'all':
        # converter.post_training_quantize = True    # True flag is used to convert float model
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]    # just quantisize weight to 8bit
        converter.inference_type = tf.uint8
        converter.quantized_input_stats = {input_arrays[0]: (0.0, 1.0)}
        if enable_dummy_quant:
            converter.default_ranges_stats = (0, 6)

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

    # Construct a copy for debug
    saver = tf.train.Saver()
    save_path_ = os.path.join(model_dir, "model-lite.ckpt")
    saver.save(sess, save_path_)

    ckpt_dir = model_dir
    ckpt_path = tf.train.latest_checkpoint(ckpt_dir)

    print(ckpt_path)

    freeze_graph.freeze_graph(input_graph=pb_txt_path, input_saver='', input_binary=False,
                              input_checkpoint=ckpt_path, output_node_names=output_node_names,
                              restore_op_name='save/restore_all', filename_tensor_name='save/Const:0',
                              output_graph=pb_output_path, clear_devices=True, initializer_nodes='')

def freeze(sess, model_dir, graph_name="frozen_graph.pb"):
    pb_output_path = os.path.join(model_dir, graph_name)
    # Turn all the variables into inline constants inside the graph and save it.
    frozen_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def, ['prediction'])
    tf.train.write_graph(
        frozen_graph_def,
        model_dir,
        graph_name,
        as_text=False)
    tf.logging.info('Saved frozen graph to %s', pb_output_path)

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

def create_inference_graph_no_mfcc(FLAGS):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Begin by making sure we have the training data we need. If you already have
    # training data of your own, use `--data_url= ` on the command line to avoid
    # downloading.
    model_settings = models.prepare_model_settings(
        len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
        FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
        FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
    # audio_processor = input_data.AudioProcessor(
    #     FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
    #     FLAGS.unknown_percentage,
    #     FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
    #     FLAGS.testing_percentage, model_settings)
    fingerprint_size = model_settings['fingerprint_size']
    # label_count = model_settings['label_count']
    # time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
    # Figure out the learning rates for each training phase. Since it's often
    # effective to have high learning rates at the start of training, followed by
    # lower levels towards the end, the number of steps and learning rates can be
    # specified as comma-separated lists to define the rate at each stage. For
    # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
    # will run 13,000 training loops in total, with a rate of 0.001 for the first
    # 10,000, and 0.0001 for the final 3,000.

    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

    logits = models.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        FLAGS.model_size_info,
        is_training=False)

    print(logits)
    tf.nn.softmax(logits, name="prediction")

def convert(sess,
            model_dir,
            FLAGS,
            inference_type="float32",
            output_node_name="logits/activation",
            enable_dummy_quant=False,
            triplet="conv-bn-relu",
            ):
    
    if FLAGS.embed_mfcc == True:
        create_inference_graph(FLAGS.wanted_words, FLAGS.sample_rate,
                            FLAGS.clip_duration_ms, FLAGS.clip_stride_ms,
                            FLAGS.window_size_ms, FLAGS.window_stride_ms,
                            FLAGS.dct_coefficient_count, FLAGS.model_architecture,
                            FLAGS.model_size_info, output_node_name)
    else:
        create_inference_graph_no_mfcc(FLAGS)


    models.load_variables_from_checkpoint(sess, FLAGS.checkpoint)

    tf.logging.info("Freezing Graph...")
    # 每次freeze后运行test_pb都会在不同的层报错说变量没有初始化，估计是变量没有顺利保存下来，可是tflite能用就很奇怪
    save_to_graphpb_and_freeze(sess,
                            model_dir=model_dir,
                            output_node_names=output_node_name)
    # freeze(sess, FLAGS.model_dir)
    tf.logging.info("Freezing complete!")

    if FLAGS.embed_mfcc == False:
        ENABLE_DUMMY_QUANT = enable_dummy_quant
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        tf.logging.info("Converting to TFLite Models...")
        convert_frozen_graph_to_tflite(input_array='fingerprint_input',
                                    output_array=output_node_name,
                                    save_path=model_dir,
                                    quantisize_type=FLAGS.quantize_type,
                                    enable_dummy_quant=ENABLE_DUMMY_QUANT)
        tf.logging.info("Convert complete! Quantize type: " + str(FLAGS.quantize_type))
        tf.logging.info("Complete!")

def main(_):
    sess = tf.InteractiveSession()
    convert(sess, model_dir=FLAGS.model_dir,
                output_node_name="prediction", FLAGS=FLAGS)
    # convert2(sess, FLAGS.model_dir, 'uint8', output_node_name='prediction')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
        '--clip_stride_ms',
        type=int,
        default=30,
        help='How often to run recognition. Useful for models with cache.',)
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is',)
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How long the stride is between spectrogram timeslices',)
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint',)
    parser.add_argument(
        '--model_dir',
        type=str,
        default='',
        help='The directory where the model is saved.')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
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
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
        help='Words to use (others will be added to an unknown label)',)
    parser.add_argument(
        '--output_file', type=str, help='Where to save the frozen graph.')
    parser.add_argument(
        '--embed_mfcc', 
        default=False,
        dest='embed_mfcc',
        action='store_true', 
        help='Embed mfcc module into graph, and the input will change.')
    parser.add_argument(
        '--quantize_type',
        type=str, 
        default=None,
        help='Quantize weight or all operations. Type: weight | all')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)