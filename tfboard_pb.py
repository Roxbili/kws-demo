#-*- encoding: utf-8 -*-

import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_path',
    type=str,
    default='mobilenetv3_quant_freeze_tflite/frozen_graph.pb',
    help='pb model path.')
parser.add_argument(
    '--summary_dir',
    type=str,
    default='mobilenetv3_quant_freeze_tflite_log',
    help='The path to save summary log.')

args = parser.parse_args()

model = args.model_path
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_def.ParseFromString(gfile.FastGFile(model, 'rb').read())
tf.import_graph_def(graph_def, name='graph')
summaryWriter = tf.summary.FileWriter(args.summary_dir, graph)

print('Save successfully.')