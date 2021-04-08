from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import input_data
import models
from tensorflow.python.platform import gfile
from tensorflow.contrib import slim as slim

from convert_to_tflite import convert
FLAGS = None

def main(_):
    # Start a new TensorFlow session.
	sess = tf.InteractiveSession()

	convert(sess, model_dir=FLAGS.train_dir,
			inference_type="uint8",
			output_node_name="prediction")


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
		'--background_volume',
		type=float,
		default=0.1,
		help="""\
      How loud the background noise should be, between 0 and 1.
      """)
	parser.add_argument(
		'--background_frequency',
		type=float,
		default=0.8,
		help="""\
      How many of the training samples have background noise mixed in.
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
		'--time_shift_ms',
		type=float,
		default=100.0,
		help="""\
      Range to randomly shift the training audio by in time.
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
		'--how_many_training_steps',
		type=str,
		default='15000,3000',
		help='How many training loops to run',)
	parser.add_argument(
		'--eval_step_interval',
		type=int,
		default=400,
		help='How often to evaluate the training results.')
	parser.add_argument(
		'--learning_rate',
		type=str,
		default='0.001,0.0001',
		help='How large a learning rate to use when training.')
	parser.add_argument(
		'--batch_size',
		type=int,
		default=100,
		help='How many items to train with at once',)
	parser.add_argument(
		'--summaries_dir',
		type=str,
		default='/tmp/retrain_logs',
		help='Where to save summary logs for TensorBoard.')
	parser.add_argument(
		'--wanted_words',
		type=str,
		default='yes,no,up,down,left,right,on,off,stop,go',
		help='Words to use (others will be added to an unknown label)',)
	parser.add_argument(
		'--train_dir',
		type=str,
		default='/tmp/speech_commands_train',
		help='Directory to write event logs and checkpoint.')
	parser.add_argument(
		'--save_step_interval',
		type=int,
		default=100,
		help='Save model checkpoint every save_steps.')
	parser.add_argument(
		'--start_checkpoint',
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
		'--check_nans',
		type=bool,
		default=False,
		help='Whether to check for invalid numbers during processing')
	parser.add_argument("--quant", action='store_true',
	                    default=False)
	parser.add_argument("--bits", type=int,
	                    default=8)


	FLAGS, unparsed = parser.parse_known_args()	# FLAGS 是一个命名空间，里面只包含第一个参数；unparsed则是剩下的参数
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
