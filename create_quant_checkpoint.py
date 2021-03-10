# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modifications Copyright 2017 Arm Inc. All Rights Reserved.           
# Added model dimensions as command line argument and changed to Adam optimizer
#
#
"""Simple speech recognition to spot a limited number of keywords.

This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio_recognition.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run tensorflow/examples/speech_commands:train

This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down

"""
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
	# We want to see all the logging messages for this tutorial.
	tf.logging.set_verbosity(tf.logging.INFO)

	# Start a new TensorFlow session.
	sess = tf.InteractiveSession()

	# Begin by making sure we have the training data we need. If you already have
	# training data of your own, use `--data_url= ` on the command line to avoid
	# downloading.
	model_settings = models.prepare_model_settings(
		len(input_data.prepare_words_list(FLAGS.wanted_words.split(','))),
		FLAGS.sample_rate, FLAGS.clip_duration_ms, FLAGS.window_size_ms,
		FLAGS.window_stride_ms, FLAGS.dct_coefficient_count)
	audio_processor = input_data.AudioProcessor(
		FLAGS.data_url, FLAGS.data_dir, FLAGS.silence_percentage,
		FLAGS.unknown_percentage,
		FLAGS.wanted_words.split(','), FLAGS.validation_percentage,
		FLAGS.testing_percentage, model_settings)
	fingerprint_size = model_settings['fingerprint_size']
	label_count = model_settings['label_count']
	time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)
	# Figure out the learning rates for each training phase. Since it's often
	# effective to have high learning rates at the start of training, followed by
	# lower levels towards the end, the number of steps and learning rates can be
	# specified as comma-separated lists to define the rate at each stage. For
	# example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
	# will run 13,000 training loops in total, with a rate of 0.001 for the first
	# 10,000, and 0.0001 for the final 3,000.
	training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
	learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
	if len(training_steps_list) != len(learning_rates_list):
		raise Exception(
			'--how_many_training_steps and --learning_rate must be equal length '
			'lists, but are %d and %d long instead' % (len(training_steps_list),
													   len(learning_rates_list)))

	fingerprint_input = tf.placeholder(
		tf.float32, [None, fingerprint_size], name='fingerprint_input')

	logits = models.create_model(
		fingerprint_input,
		model_settings,
		FLAGS.model_architecture,
		FLAGS.model_size_info,
		is_training=False)

	print(logits)

	predictions = tf.nn.softmax(logits, name="prediction")

	# Define loss and optimizer
	ground_truth_input = tf.placeholder(
		tf.float32, [None, label_count], name='groundtruth_input')

	# Optionally we can add runtime checks to spot when NaNs or other symptoms of
	# numerical errors start occurring during training.
	control_dependencies = []
	if FLAGS.check_nans:
		checks = tf.add_check_numerics_ops()
		control_dependencies = [checks]

	if FLAGS.quant:
		tf.logging.info("Adding quantization ops and creating eval graph...")
		tf.contrib.quantize.experimental_create_eval_graph(sess.graph,
		                                                   weight_bits=FLAGS.bits,
		                                                   activation_bits=FLAGS.bits,
		                                                   quant_delay=0)
		print("Done")

	if FLAGS.start_checkpoint:
		models.load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)


	predicted_indices = tf.argmax(logits, 1)
	expected_indices = tf.argmax(ground_truth_input, 1)
	correct_prediction = tf.equal(predicted_indices, expected_indices)
	confusion_matrix = tf.confusion_matrix(
		expected_indices, predicted_indices, num_classes=label_count)
	evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	set_size = audio_processor.set_size('testing')
	tf.logging.info('set_size=%d', set_size)
	total_accuracy = 0
	total_conf_matrix = None
	for i in xrange(0, set_size, FLAGS.batch_size):
		test_fingerprints, test_ground_truth = audio_processor.get_data(
			FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
		test_accuracy, conf_matrix = sess.run(
			[evaluation_step, confusion_matrix],
			feed_dict={
				fingerprint_input: test_fingerprints,
				ground_truth_input: test_ground_truth,
			})
		batch_size = min(FLAGS.batch_size, set_size - i)
		total_accuracy += (test_accuracy * batch_size) / set_size
		if total_conf_matrix is None:
			total_conf_matrix = conf_matrix
		else:
			total_conf_matrix += conf_matrix
	tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
	tf.logging.info('Final test accuracy = %.2f%% (N=%d)' % (total_accuracy * 100,
															 set_size))

	# Finally, save the model.
	saver = tf.train.Saver()
	save_path = os.path.join(FLAGS.train_dir, "model-final")
	saver.save(sess, save_path)

	np_dict = {}

	save_dir = os.path.join(FLAGS.train_dir, "weights_npy")

	for var in tf.all_variables():
		np_dict[var.name] =  var.eval()

	np.save(save_dir, np_dict)

	if FLAGS.bits == 8:
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


