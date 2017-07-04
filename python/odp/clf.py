import tensorflow as tf
import tflearn
import numpy as np

def single_classifier_train(X_train, y_train, num_epoch=100):

    num_train, dim = X_train.shape
    num_classes = len(np.unique(y_train))
    # clear tf variables
    tf.reset_default_graph()
    tflearn.config.init_graph(seed=None, log_device=False, gpu_memory_fraction=0)
    input_data = tflearn.input_data(shape=[None, dim])
    net = tflearn.layers.core.fully_connected(input_data, num_classes, activation="Softmax")
    acc = tflearn.metrics.Accuracy()
    regression = tflearn.regression(net,
                                    dtype=tf.float64,
                                    metric=acc,
                                    to_one_hot=True,
                                    n_classes=num_classes,
                                    learning_rate=0.001,
                                    batch_size=600,
                                    validation_batch_size=30)


    m = tflearn.DNN(regression, tensorboard_verbose=3)
    m.fit(X_train, y_train, n_epoch=num_epoch, show_metric=True)

    return m

def single_classifier_predict(X_test, M, top_k=1):
    probs = M.predict(X_test)

    if (top_k > 1):
        y_pred = np.argsort(probs, axis = 1)[:, -top_k:]
    else:
        y_pred = np.argmax(probs, axis = 1)
    return y_pred
