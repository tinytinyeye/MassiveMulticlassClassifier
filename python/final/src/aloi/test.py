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
        f = open('./result.out', 'a+')
        print("running multi")
        B = 15
        for R in [50, 70]:
            if (B * R > 1000):
                continue
            else:
                print("Computing B=%d R=%d" % (B, R))
                clf = MultiClassifier(R, B)
                clf.train(X_train, y_train, n_epoch=100)
                clf.fine_tune(X_train, y_train, num_epoch=150)
                y_pred_avg = clf.predict_with_average(X_test, k)
                y_pred_weight = clf.predict_with_weight(X_test, k)
                print("====================================")
                acc_avg = clf.calc_accuracy(y_pred_avg, y_test)
                acc_weight = clf.calc_accuracy(y_pred_weight, y_test)
                print("B = %d R = %d acc_avg = %.4f" % (B, R, acc_avg))
                print("B = %d R = %d acc_weight = %.4f" % (B, R, acc_weight))
                f.write("B = {} R = {} acc_avg = {:.4f}\n".format(B, R, acc_avg))
                f.write("B = {} R = {} acc_weight = {:.4f}\n".format(B, R, acc_weight))
                f.write("W = {}\n".format(str(clf.get_weight())))
                f.write("====================================\n")
                print("====================================")
        for B in range(30, 100, 10):
            for R in range(10, 55, 5):
                if (B * R > 1000):
                    continue
                else:
                    print("Computing B=%d R=%d" % (B, R))
                    clf = MultiClassifier(R, B)
                    clf.train(X_train, y_train, n_epoch=60)
                    clf.fine_tune(X_train, y_train, num_epoch=300)
                    y_pred_avg = clf.predict_with_average(X_test, k)
                    y_pred_weight = clf.predict_with_weight(X_test, k)
                    print("====================================")
                    acc_avg = clf.calc_accuracy(y_pred_avg, y_test)
                    acc_weight = clf.calc_accuracy(y_pred_weight, y_test)
                    print("B = %d R = %d acc_avg = %.4f" % (B, R, acc_avg))
                    print("B = %d R = %d acc_weight = %.4f" % (B, R, acc_weight))
                    f.write("B = {} R = {} acc_avg = {:.4f}\n".format(B, R, acc_avg))
                    f.write("B = {} R = {} acc_weight = {:.4f}\n".format(B, R, acc_weight))
                    f.write("W = {}\n".format(str(clf.get_weight())))
                    f.write("====================================\n")
                    print("====================================")
        f.close()
    else:
        print("running single")
        M = single_classifier_train(X_train, y_train)
        print("train finished")
        y_pred = single_classifier_predict(X_test, M, multi_eval=False)
        print("predict finished")
        print("accuracy is ", calc_accuracy(y_pred, y_test, multi_eval=False))


if __name__ == "__main__":
    main()
