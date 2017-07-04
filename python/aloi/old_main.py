import numpy as np
from sklearn import linear_model
from data_util import *
from hash import *

multi = True

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

def multi_classifier_train_and_predict_mapped():
    data = map_and_split(20)
    for i in range(20):
        X_train = data[i][0]
        y_train = data[i][1]
        X_test = data[i][2]
        y_test = data[i][3]
        M = linear_model.LogisticRegression(n_jobs=-1)
        M.fit(X_train, y_train)
        P = M.predict_proba(X_test)
        y_pred = np.argmax(P, axis = 1)
        acc = calc_accuracy(y_pred, y_test)
        print("model %d get accuracy %f" % (i, acc))


def multi_classifier_predict(X, R, B, M, a, b, p=105943, SGD=False):
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
        r = np.zeros((R, num_test))
        for j in range(R):
            pj = P[j]
            c = H(i, a[j], b[j], B, p)
            column = pj[:, c]
            r[j] = column
        # GP[:, i] = np.amin(r, axis=0)
        GP[:, i] = np.mean(r, axis=0)
        # GP[:, i] = np.median(r, axis=0)

    y_pred = np.argmax(GP, axis = 1)
    return y_pred

def multi_classifier_individual_score(M, X_test, y_test, a, b, p, R, B):
    print("in predict individual")
    num_test = X_test.shape[0]
    k = X_test.shape[1]
    P = []
    for i in range(R):
        probs = M[i].predict_proba(X_test)
        y_pred = np.argmax(probs, axis = 1)
        mapper = lambda x: ((a[i] * x + b[i]) % p) % B
        vfunc = np.vectorize(mapper)
        y_i = vfunc(y_test)
        acc = calc_accuracy(y_pred, y_i)
        print("model %d get accuracy %f" % (i, acc))

def single_classifier_train(X_train, y_train, SGD=False):
    M = linear_model.LogisticRegression(n_jobs=-1)
    if SGD:
        M = linear_model.SGDClassifier(loss='log', n_iter=1000, n_jobs=-1, verbose=True)
    M.fit(X_train, y_train)
    return M

def single_classifier_predict(X_test, M):
    probs = M.predict_proba(X_test)
    y_pred = np.argmax(probs, axis = 1)
    return y_pred

def calc_accuracy(y_test_pred, y_test):
    test_accuracy = np.mean(y_test == y_test_pred)
    return test_accuracy

def main():
    X_train, y_train, X_test, y_test = get_aloi_data_saved()
    print("load finished")
    if multi:
        R = 1
        for B in [250, 300, 400, 500]:
            # for R in range(2, 6):
            if (B * R > 1000):
                continue
            else:
                print("Computing R=%d B=%d" % (R, B))
                a, b, p, M = multi_classifier_train(X_train, y_train, R, B)
                y_pred = multi_classifier_predict(X_test, R, B, M, a, b)
                print("====================================")
                print("R=%d B=%d acc=%f" % (R, B, calc_accuracy(y_pred, y_test)))
                print("====================================")
    else:
        M = single_classifier_train(X_train, y_train, SGD=False)
        print("train finished")
        y_pred = single_classifier_predict(X_test, M)
        print("predict finished")
        print("accuracy is ", calc_accuracy(y_pred, y_test))


if __name__ == "__main__":
    main()
