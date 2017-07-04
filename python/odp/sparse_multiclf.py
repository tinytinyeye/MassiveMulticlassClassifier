import sys,os
import time
import tensorflow as tf
from hash import *

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('num_epochs', 5, 'Number of epochs to run trainer.')
flags.DEFINE_integer('num_preprocess_threads', 12, '')

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

MIN_AFTER_DEQUEUE = 10000

NUM_FEATURES = 422713
NUM_CLASSES = 105033

def matmul(X, W):
    """
    general purpose matrix multiplication
    params:
    X - dense tensor or sparse tensor in the form of (indices, values)
    W - weights
    """
    if type(X) == tf.Tensor:
        return tf.matmul(X, W)
    else:
        return tf.nn.embedding_lookup_sparse(W, X[0], X[1], combiner="sum")

def init_weights(shape, stddev=0.01, name=None):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)

def init_bias(shape, val=0.0, name=None):
    init = tf.constant(val, shape=shape)
    return tf.Variable(init, name=name)

def decode(batch_serialized_examples):
    features = tf.parse_example(
        batch_serialized_examples,
        features={
            'label' : tf.FixedLenFeature([], tf.int64),
            'index' : tf.VarLenFeature(tf.int64),
            'value' : tf.VarLenFeature(tf.float32),
        }
    )
    labels = features['label']
    indices = features['index']
    values = features['value']

    return labels, indices, values

class Timer():
  def __init__(self):
    self.start_time = time.time()

  def elapsed(self):
    end_time = time.time()
    duration = end_time - self.start_time
    self.start_time = end_time
    return duration

class MultiClassifier(object):
    def __init__(self,
                    R,
                    B,
                    num_features,
                    num_classes,
                    p=105943,
                    hash_save_path=None
                    restore_path=None,
                    model_path=None
                    checkpoint_path=None):
        """
        M - list of saved models, need to use saver.restore
        R - number of models
        B - number of classes in each model
        p - prime number used in hashing
        a - parameters used in hashing
        b - parameters used in hashing
        num_features - number of features in dataset
        num_classes - number of classes in dataset
        restore_path - path to the saved hash parameters
        model_path - path to saved weights and bias for all models
        checkpoint_dir - path to directory that saves inidividual weight and
                            bias for each model
        """
        self.weights = [] # trained weights for each model
        self.bias = [] # trained bias for each model
        self.num_features = num_features
        self.num_classes = num_classes
        self.restore_path = restore_path
        self.model_path = model_path
        self.checkpoint_dir = checkpoint_dir
        self.W = None
        self.R = R
        self.B = B
        self.p = p
        self.a = None
        self.b = None

        if hash_save_path is None:
            path = "./model"
        # if no hash parameters are recovered, save new parameters to given path
        # need to provide hash_save_path for the first time
        if self.restore_path is None:
            h_a, h_b = get_hash_params(self.num_classes, self.R, self.B)
            self.a = np.array(h_a)
            self.b = np.array(h_b)
            np.savez(hash_save_path, a=h_a, b=h_b)
        else:
            hash_params = np.load(restore_path)
            self.a = hash_params['a']
            self.b = hash_params['b']
        if model_path is not None:
            params = np.load(model_path)
            self.weights = params['weights']
            self.bias = params['bias']

    def load_sparse(self,
                    filename,
                    graph,
                    batch_size=FLAGS.batch_size,
                    num_epochs=FLAGS.num_epochs):
        """
        load sparse matrix data from tfrecord file and save to X and y
        """
        with graph.as_default():
            filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=num_epochs)
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            batch_serialized_examples = tf.train.shuffle_batch(
                [serialized_example],
                batch_size=batch_size,
                num_threads=FLAGS.num_preprocess_threads,
                capacity=MIN_AFTER_DEQUEUE+(FLAGS.num_preprocess_threads+1)*batch_size,
                min_after_dequeue=MIN_AFTER_DEQUEUE
            )

            return decode(batch_serialized_examples)

    def train_single(self, filename, clf_id, num_epochs=5):
        """
        filename - input file, in the format of TFRecords
        clf_id - classifier id, used to identify which hash parameter to use
        num_epochs - number of epochs running for each classifier
        """
        # return value
        ret_weight = None
        ret_bias = None
        graph = tf.Graph()
        with graph.as_default():
            # build input pipeline
            labels, indices, values = self.load_sparse(filename,
                                                        graph,
                                                        num_epochs=num_epochs)
            X = (indices, values)
            y = labels
            # hash labels to the parameters for given classifier
            y_h = tf.map_fn(hash_factory(self.a[clf_id],
                                         self.b[clf_id],
                                         self.B), y)
            W = init_weights([num_features, self.B], name="weights")
            b = init_bias([self.B], name="bias")

            # build graph
            y_p = matmul(X, W) + b
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=y_p,
                                    labels=y_h)
                                    )
            prediction = tf.nn.in_top_k(y_p, y_h, 1)
            accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
            train_op = tf.train.AdamOptimizer().minimize(loss)
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            sess = tf.Session()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
              step = 0
              timer = Timer()
              while not coord.should_stop():
                  _, loss_, accuracy_ = sess.run([train_op, loss, accuracy])
                  if step % 100 == 0:
                      print('step:', step,
                            'train precision@1:', accuracy_,
                            'loss:', loss_,
                            'duration:', timer.elapsed())
                  step += 1
            except tf.errors.OutOfRangeError:
                print('Done training for %d epochs, %d steps.' %
                        (num_epochs, step))
            finally:
                coord.request_stop()

            ret_weight = sess.run(W)
            ret_bias = sess.run(b)
            # self.weights.append(sess.run(W))
            # self.bias.append(sess.run(b))
            # if self.model_path is None:
            #     w_ = np.array(self.weights)
            #     b_ = np.array(self.bias)
            #     np.savez("./model", weights=w_, bias=b_)

            coord.join(threads)
            sess.close()

            return ret_weight, ret_bias

    def train(self, filename, num_epochs=5, resume_from=0):
        """
        filename - input file, in the format of TFRecords
        num_epochs - number of epochs running for each classifier
        resume_from - resume training from a specific classifier, require
                        restore path and checkpoint directory
        """
        if resume_from != 0 and (restore_path is None or checkpoint_dir is None):
            raise ValueError, "restore path and checkpoint dir are required to resume"

        # load checkpoints into memory before continue
        if resume_from > 0:
            for i in range(start_from):
                params = np.load(model_path)
                w = params['weight']
                b = params['bias']
                self.weights.append(w)
                self.bias.append(b)

        for i in range(resume_from, self.R):
            weight, bias = self.train_single(filename, i, num_epochs=num_epochs)
            self.weights.append(weight)
            self.bias.append(bias)
            # if self.model_path is None:
            #     w_ = np.array(self.weights)
            #     b_ = np.array(self.bias)
            #     np.savez("./model", weights=w_, bias=b_)


    def predict(self, filename, num_features=None, num_classes=None):
        if num_features == None or num_classes == None:
            num_features = NUM_FEATURES
            num_classes = NUM_CLASSES
        graph = tf.Graph()
        # saver = tf.train.Saver()
        with graph.as_default():
            # build input pipeline
            labels, indices, values = self.load_sparse(filename,
                                                        graph,
                                                        batch_size=10000,
                                                        num_epochs=1)
            X = (indices, values)
            y = labels
            y_h = tf.map_fn(hash_factory(self.a[0], self.b[0], self.B), y)
            W = tf.constant(self.weights[0])
            b = tf.constant(self.bias[0])
            # build graph
            y_p = matmul(X, W) + b
            prediction = tf.nn.in_top_k(y_p, y_h, 1)
            correct_prediction = tf.reduce_sum(tf.cast(prediction, tf.int32))
            num_prediction = tf.shape(prediction)[0]
            # accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())
            sess = tf.Session()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            total_accuracy = 0.0
            correct = 0
            total = 0
            try:
              step = 0
              timer = Timer()
              while not coord.should_stop():
                correct_ = sess.run(correct_prediction)
                total_ = sess.run(num_prediction)
                correct += correct_
                total += total_
            except tf.errors.OutOfRangeError:
                print("Done predicting")
            finally:
                coord.request_stop()
            coord.join(threads)
            print("accuracy is", correct / total)
            sess.close()


def main(_):
    clf = MultiClassifier(R=100, B=100, restore_path="./hash_params.npz", model_path="model.npz")
    # clf.train(filename="./data/tfrecords/train.tfrecords", num_epochs=1)
    clf.predict(filename="./data/tfrecords/test.tfrecords")

if __name__ == '__main__':
  tf.app.run()
