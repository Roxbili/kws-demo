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
    q = r * std_dev + mean_
    q = q.astype(np.uint8)
    return q

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

    fingerprint_input = tf.placeholder(
        tf.float32, [None, fingerprint_size], name='fingerprint_input')

    logits = models.create_model(
        fingerprint_input,
        model_settings,
        FLAGS.model_architecture,
        FLAGS.model_size_info,
        is_training=False)

    ground_truth_input = tf.placeholder(
        tf.float32, [None, label_count], name='groundtruth_input')

    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.confusion_matrix(
        expected_indices, predicted_indices, num_classes=label_count)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=FLAGS.tflite_path)
    # interpreter = tf.lite.Interpreter(model_path="tflite_factory/swiftnet-uint8.lite")
    interpreter.allocate_tensors()

    
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


    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)