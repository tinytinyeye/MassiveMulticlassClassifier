import numpy as np
import tensorflow as tf
import numpy as np
import tflearn
import os
from data_util import *
from hash import *

multi = False

def calc_accuracy(y_test_pred, y_test, multi_eval=False):
    test_accuracy = 0
    if (not multi_eval):
        test_accuracy = np.mean(y_test == y_test_pred)
    else:
        correct = 0
        total = y_test.shape[0]
        for i in range(total):
            if y_test[i] in y_test_pred[i]:
                correct += 1
        print("{}/{}".format(correct, total))
        test_accuracy = correct / total
    return test_accuracy

def single_classifier_train(X_train, y_train, num_epoch=100):

    num_train, dim = X_train.shape
    num_classes = len(np.unique(y_train))
    # TODO:tflearn
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
    print("X shape: {} y shape {}".format(X_train.shape, y_train.shape))
    m.fit(X_train, y_train, n_epoch=num_epoch, show_metric=True)

    return m

def single_classifier_predict(X_test, M, multi_eval=False):
    probs = M.predict(X_test)

    if (multi_eval):
        y_pred = np.argsort(probs, axis = 1)[:, -3:]
    else:
        y_pred = np.argmax(probs, axis = 1)
    return y_pred

def multi_classifier_train(X_train, y_train, R, B, p=105943):
    M = []
    a, b, y = hash_and_test(y_train, R, B, p)
    print("y length is", len(y))
    for i in range(R):
        m = single_classifier_train(X_train, y[i], num_epoch=5)
        M.append(m)
        print("model %d taining finished" % (i))
    return a, b, p, M

def multi_classifier_predict(X, R, B, M, a, b, p=105943, multi_eval=False):
    num_test = X.shape[0]
    k = 1000
    GP = np.zeros((num_test, k)) # global probabilities for each class
    P = []
    for i in range(R):
        probs = M[i].predict(X)
        P.append(probs) # P[i] is (n_test * B)

    for i in range(k):
            # aggregate with average
            # r is a matrix that contains result of the following:
            # for class i in classifier j, the probability for class
            # h_j(i)
            probs_i = np.zeros((num_test, R))
            for j in range(R):
                probs_i[:, j] = P[j][:, H(i, a[j], b[j], B, p)]
            GP[:, i] = np.mean(probs_i, axis=1)

    if (not multi_eval):
        y_pred = np.argmax(GP, axis = 1)
    else:
        y_pred = np.argsort(GP, axis = 1)[:, -3:]
    return y_pred

def main():
    (num_points, num_features, num_labels) = (1084404, 421705, 105033)
    print("odp data contains %d points, %d features and %d labels" %
                                (num_points, num_features, num_labels))
    # X_train, y_train = load_odp_data_raw("./data/odp_train.vw.gz", num_features)
    # X_test, y_test = load_odp_data_raw("./data/odp_test.vw.gz", num_features)
    X_train, y_train = get_odp_train_data(num_features, sample_size=0.1)
    save_odp_data("train", X_train, y_train)
    X_test, y_test = get_odp_test_data(num_features, sample_size=0.1)
    save_odp_data("test", X_test, y_test)
    print("running single")
    M = single_classifier_train(X_train, y_train, SGD=False)
    print("train finished")
    y_pred = single_classifier_predict(X_test, M, multi_eval=True)
    print("predict finished")
    accuracy = calc_accuracy(y_pred, y_test, multi_eval=True)
    print("accuracy is ", accuracy)
    with open('./result.out', 'w') as f:
        f.write(accuracy)
    # X_train, y_train, X_test, y_test = get_aloi_data_saved()
    # print("load finished")
    # if multi:
    #     for B in [50, 75, 100, 150, 200, 300]:
    #         for R in range(3, 16, 3):
    #             if (B * R > 1000):
    #                 continue
    #             else:
    #                 print("Computing B=%d R=%d" % (B, R))
    #                 a, b, p, M = multi_classifier_train(X_train, y_train, R, B)
    #                 y_pred = multi_classifier_predict(X_test, R, B, M, a, b,
    #                                         multi_eval=True, dynamic_eval=True)
    #                 print("====================================")
    #                 print("B=%d R=%d acc=%f" % (B, R, calc_accuracy(y_pred, y_test, multi_eval=True)))
    #                 print("====================================")
    # else:
    #     print("running single")
    #     M = single_classifier_train(X_train, y_train, SGD=False)
    #     print("train finished")
    #     y_pred = single_classifier_predict(X_test, M, multi_eval=True)
    #     print("predict finished")
    #     print("accuracy is ", calc_accuracy(y_pred, y_test, multi_eval=True))


if __name__ == "__main__":
    main()
