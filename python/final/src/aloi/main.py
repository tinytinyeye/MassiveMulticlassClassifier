from multiclf import *
import numpy as np
import os

multi = True

def get_aloi_data_saved():
    X_train = np.loadtxt("X_train")
    y_train = np.loadtxt("y_train", dtype=np.int32)
    X_test = np.loadtxt("X_test")
    y_test = np.loadtxt("y_test", dtype=np.int32)
    return X_train, y_train, X_test, y_test

def main():
    X_train, y_train, X_test, y_test = get_aloi_data_saved()
    k = len(np.unique(y_test))
    print("load finished")
    if multi:
        log = open('./result.out', 'w')
        print("running multi")
        for B in [50, 100, 150, 200, 300]:
            for R in range(5, 16, 3):
                if (B * R > 1000):
                    continue
                else:
                    print("Computing B=%d R=%d" % (B, R))
                    clf = MultiClassifier(R, B)
                    clf.train(X_train, y_train, n_epoch=5)
                    y_pred = clf.predict_with_average(X_test, k)
                    print("====================================")
                    acc = clf.calc_accuracy(y_pred, y_test)
                    print("B=%d R=%d acc=%f" % (B, R, acc))
                    log.write("B={} R={} acc={}\n".format(B, R, acc))
                    print("====================================")
        log.close()
    else:
        print("running single")
        M = single_classifier_train(X_train, y_train)
        print("train finished")
        y_pred = single_classifier_predict(X_test, M, multi_eval=False)
        print("predict finished")
        print("accuracy is ", calc_accuracy(y_pred, y_test, multi_eval=False))


if __name__ == "__main__":
    main()
