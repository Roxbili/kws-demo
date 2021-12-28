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
    


    ########################### save dataset ###########################
    # training set
    set_size = audio_processor.set_size('training')
    print('set_size=%d', set_size)
    train_save_path = os.path.join(args.data_dir, 'zynq_mfcc', 'train')
    for i in range(0, set_size, args.batch_size):
        start_time = time.time()
        training_fingerprints, training_ground_truth = audio_processor.get_data(
            args.batch_size, i, model_settings, 0.0, 0.0, 0, 'training')
        # print(test_fingerprints.shape)  # (batch_size 490)
        # print(test_ground_truth.shape)  # (batch_size, 12)
        sample_path = os.path.join(train_save_path, '{:0>5d}_{}'.format(i, words_list[training_ground_truth.argmax()]))
        np.save(sample_path, training_fingerprints)
        end_time = time.time()
        print('Save path: {}, time: {}'.format(sample_path, end_time - start_time))
        

    # validation set
    set_size = audio_processor.set_size('validation')
    print('set_size=%d', set_size)
    val_save_path = os.path.join(args.data_dir, 'zynq_mfcc', 'val')
    for i in range(0, set_size, args.batch_size):
        start_time = time.time()
        validation_fingerprints, validation_ground_truth = audio_processor.get_data(
            args.batch_size, i, model_settings, 0.0, 0.0, 0, 'validation')
        # print(test_fingerprints.shape)  # (batch_size 490)
        # print(test_ground_truth.shape)  # (batch_size, 12)
        sample_path = os.path.join(val_save_path, '{:0>5d}_{}'.format(i, words_list[validation_ground_truth.argmax()]))
        np.save(sample_path, validation_fingerprints)
        end_time = time.time()
        print('Save path: {}, time: {}'.format(sample_path, end_time - start_time))
    

    # test set
    set_size = audio_processor.set_size('testing')
    print('set_size=%d', set_size)
    test_save_path = os.path.join(args.data_dir, 'zynq_mfcc', 'test')
    for i in range(0, set_size, args.batch_size):
        start_time = time.time()
        test_fingerprints, test_ground_truth = audio_processor.get_data(
            args.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing')
        # print(test_fingerprints.shape)  # (batch_size 490)
        # print(test_ground_truth.shape)  # (batch_size, 12)
        sample_path = os.path.join(test_save_path, '{:0>5d}_{}'.format(i, words_list[test_ground_truth.argmax()]))
        np.save(sample_path, test_fingerprints)
        end_time = time.time()
        print('Save path: {}, time: {}'.format(sample_path, end_time - start_time))


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

    # 设置
    args.model_architecture = 'mobilenet-v3'
    args.dct_coefficient_count = 10
    args.batch_size = 1
    args.window_size_ms = 40
    args.window_stride_ms = 20
    args.model_size_info = [4, 16, 10, 4, 2, 2, 16, 3, 3, 1, 1, 2, 32, 3, 3, 1, 1, 2, 32, 5, 5, 1, 1, 2]
    args.testing_mode = 'simulate'
    args.tflite_path = 'test_log/mobilenetv3_quant_mfcc_gen/symmetric_8bit_mean220_std0.97.lite'
    args.data_dir = 'speech_dataset'

    main(args)
