# 14629 | 2708:0.235702 8021:0.235702 11947:0.235702 46262:0.235702 55402:0.235702 95222:0.235702 106077:0.2357\
# 02 148503:0.235702 166122:0.235702 167359:0.235702 204459:0.235702 215918:0.235702 268910:0.235702 302013:0.2\
# 35702 329485:0.235702 349947:0.235702 361878:0.235702 422712:1

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

import numpy as np
import gzip

_float_feature = lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v))
_int_feature = lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def main(argv):
  writer = tf.python_io.TFRecordWriter(argv[2])
  num_points = 0
  with gzip.open(argv[1], 'r') as f:
      lines = [x.decode('utf8').strip() for x in f.readlines()]
      for line in lines:
          data = line.split('|')
          label = int(data[0].strip())
          features_str = data[1].strip()
          features = features_str.split(' ')

          indices = []
          values = []

          for feature in features:
              index, value = feature.split(':')
              indices.append(int(index))
              values.append(float(value))

          label_ = _int_feature([label])
          example = tf.train.Example(
                features=tf.train.Features(
                feature={
                    'label': label_,
                    'index': _int_feature(indices),
                    'value': _float_feature(values)
                }))
          writer.write(example.SerializeToString())
          num_points += 1
          if num_points % 10000 == 0:
              print('%d lines done' % num_points)


if __name__ == '__main__':
  tf.app.run()
