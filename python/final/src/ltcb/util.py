import numpy as np
from pathlib import Path
import os
import random
import string
import time

def scan_vw_hash(vw_hash):
    """
    scan output file produced by vw to get infomation about hashed features
    """
    feature_dict = {}

    with open(vw_hash) as f:
        for line in f:
            line_ = line.split()
            if len(line_) == 1:
                continue
            features = {}
            for feature_str in line_:
                index = int(feature_str.split(':')[1])
                feature_dict[int(index)] = 1

    sorted_feature = sorted(feature_dict.keys())

    min_feature = sorted_feature[0]
    max_feature = sorted_feature[-1]
    num_features = max_feature + 1
    print(num_features, "features, min", min_feature, "max", max_feature)
    print("done")

def save_to_tfrecords(vw, vw_hash, save_path):
    """
    save dataset to tfrecord, with format (index, value, label)

    parameters:
    vw - input dataset, in the format of .vw
    vw_hash - hashed features processed by vw
    save_path - the path you save tfrecords to
    """
    label_list = []
    vw_count = 0
    with open(vw) as f:
        for line in f:
            data = line.split('|')
            # label starts from 1
            label = int(data[0].strip()) - 1
            label_list.append(label)
            vw_count += 1
            if vw_count % 10000 == 0:
                print("vw counts", vw_count)

    _float_feature = lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v))
    _int_feature = lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    writer = tf.python_io.TFRecordWriter(save_path)

    vw_hash_count = 0
    with open(vw_hash) as f:
        for line in f:
            line_ = line.split()
            if len(line_) == 1:
                continue
            features = {}
            for feature_str in line_:
                index = int(feature_str.split(':')[1])
                value = int(feature_str.split(':')[2])
                if index not in features:
                    features[index] = value
                else:
                    features[index] += value

            indices = []
            values = []

            for index, value in features.items():
                indices.append(int(index))
                values.append(float(value))

            label_ = _int_feature([label_list[vw_hash_count]])
            example = tf.train.Example(
                  features=tf.train.Features(
                  feature={
                      'label': label_,
                      'index': _int_feature(indices),
                      'value': _float_feature(values)
                  }))
            writer.write(example.SerializeToString())

            vw_hash_count += 1
            if vw_hash_count % 10000 == 0:
                print("vw hash counts", vw_hash_count)
    assert(vw_count == vw_hash_count)

def id_generator(size=3, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

class Timer():
  def __init__(self):
    self.start_time = time.time()

  def elapsed(self):
    end_time = time.time()
    duration = end_time - self.start_time
    self.start_time = end_time
    return duration
