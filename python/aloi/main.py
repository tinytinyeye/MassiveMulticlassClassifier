import numpy as np
from sklearn import linear_model
from data_util import *
from hash import *

multi = False

def multi_classifier_train(X_train, y_train, R, B, p=105943):
    """
    parameters:
    X_train: training data
    y_train: labels corresponding to training data
    R: number of classifiers
    B: number of classes in each classifier
    """
    M = []
    a, b, y = hash_and_test(y_train, R, B, p)
    for i in range(R):
        M.append(linear_model.LogisticRegression(n_jobs=-1))
        M[i].fit(X_train, y[i])
        print("model %d taining finished" % (i))
    return a, b, p, M

def multi_classifier_predict(X, R, B, M, a, b, p=105943, SGD=False,
                                multi_eval=False, dynamic_eval=False):
    """
    parameters:
    X: test data
    R: number of repetition
    B: classes in each individual classifiers
    M: model_selection
    a: hash function parameters
    b: hash function parameters
    p: hash function parameters
    SGD: use gradient descent or not
    multi_eval: choose top-3 as prediction or not
    dynamic_eval: use the

    return:
    y_pred: label prediction for X
    """
    num_test = X.shape[0]
    k = 1000
    GP = np.zeros((num_test, k)) # global probabilities for each class
    P = []
    for i in range(R):
        probs = M[i].predict_proba(X)
        P.append(probs) # P[i] is (n_test * B)

    for i in range(k):
        # aggregate with average
        if (not dynamic_eval):
            r = np.zeros((R, num_test))
            for j in range(R):
                pj = P[j]
                c = H(i, a[j], b[j], B, p)
                column = pj[:, c]
                r[j] = column
            GP[:, i] = np.mean(r, axis=0)
        # aggregate with top-3 count
        else:
            r = np.zeros((R, num_test))
            for j in range(R):
                # first get the order of the items and then get the rank
                # if the rank number is larger, it means better ranking
                ranks = P[j].argsort(axis=1).argsort(axis=1)
                c = H(i, a[j], b[j], B, p)
                column = ranks[:, c] # num_test * 1
                r[j] = column
            # print(r.T)
            GP[:, i] = np.sum(r, axis=0) # R * num_test

    if (not multi_eval):
        y_pred = np.argmin(GP, axis = 1)
    else:
        # y_pred = np.argsort(GP, axis = 1)[:, 0:3]
        y_pred = np.argsort(GP, axis = 1)[:, -3:]
    return y_pred

def single_classifier_train(X_train, y_train, SGD=False):
    M = linear_model.LogisticRegression(n_jobs=-1)
    if SGD:
        M = linear_model.SGDClassifier(loss='log', n_iter=1000, n_jobs=-1, verbose=True)
    M.fit(X_train, y_train)
    return M

def single_classifier_predict(X_test, M, multi_eval=False):
    probs = M.predict_proba(X_test)
    if (multi_eval):
        y_pred = np.argsort(probs, axis = 1)[:, -3:]
    else:
        y_pred = np.argmax(probs, axis = 1)
    return y_pred

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

def main():
    X_train, y_train, X_test, y_test = get_aloi_data_saved()
    print("load finished")
    log = open('./result.out', 'w')
    if multi:
        B = 50
        for R in range(11, 16, 3):
            print("Computing B=%d R=%d" % (B, R))
            a, b, p, M = multi_classifier_train(X_train, y_train, R, B)
            y_pred = multi_classifier_predict(X_test, R, B, M, a, b,
                                    multi_eval=True, dynamic_eval=True)
            print("====================================")
            acc = calc_accuracy(y_pred, y_test, multi_eval=True)
            print("B=%d R=%d acc=%f" % (B, R, acc))
            log.write("B={} R={} acc={}\n".format(B, R, acc))
            print("====================================")

        for B in [75, 100, 150, 200, 300]:
            for R in range(5, 16, 3):
                if (B * R > 1000):
                    continue
                else:
                    print("Computing B=%d R=%d" % (B, R))
                    a, b, p, M = multi_classifier_train(X_train, y_train, R, B)
                    y_pred = multi_classifier_predict(X_test, R, B, M, a, b,
                                            multi_eval=True, dynamic_eval=True)
                    print("====================================")
                    acc = calc_accuracy(y_pred, y_test, multi_eval=True)
                    print("B=%d R=%d acc=%f" % (B, R, acc))
                    log.write("B={} R={} acc={}\n".format(B, R, acc))
                    print("====================================")
        log.close()
    else:
        print("running single")
        M = single_classifier_train(X_train, y_train, SGD=False)
        print("train finished")
        y_pred = single_classifier_predict(X_test, M, multi_eval=True)
        print("predict finished")
        print("accuracy is ", calc_accuracy(y_pred, y_test, multi_eval=True))


if __name__ == "__main__":
    main()
