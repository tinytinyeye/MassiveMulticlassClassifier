from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
from hash import *

def load_aloi_data():
    with open("./data.txt", 'r') as f:
        meta = f.readline().split(' ')
        num_points = int(meta[0])
        num_features = int(meta[1])
        num_labels = int(meta[2])
        print("aloi data contains %d points, %d features and %d labels" % (num_points, num_features, num_labels))
        # build a mapping from label to a list of data
        label_to_data = {}
        for line in f:
            x_row = np.zeros(num_features)
            data = line.split(' ')
            label = int(data[0])
            for i in range(1, len(data) - 1):
                feature = data[i].split(':')
                x_row[int(feature[0]) - 1] = float(feature[1])
            if label not in label_to_data:
                label_to_data[label] = []
            label_to_data[label].append(x_row)
        return label_to_data

def get_aloi_data(sample_size=1, test_size=0.5):
    """
    parameters:
        sample_size: from the total data choose sample_size porportion of data
        test_size: from the data in each label, choose test_size ratio of the
        data as test data
    returns:
        X_train: training data
        y_train: labels corresponding to training data
        X_test: testing data
        y_test: labels corresponding to training data
    """
    label_to_data = load_aloi_data()
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for label in label_to_data:
        data = label_to_data[label]
        # first shuffle data for a label
        data = shuffle(data)
        # trunck sample size
        data = data[:int(len(data) * sample_size)]
        training = data[:int(len(data) * (1.0 - test_size))]
        testing = data[len(training):]
        X_train.extend(training)
        y_train.extend([label]*len(training))
        X_test.extend(testing)
        y_test.extend([label]*len(testing))
    # shuffle data before
    X_train, y_train = shuffle(X_train, y_train)
    X_train = np.array(X_train)
    y_train = np.array(y_train, np.int32)
    X_test = np.array(X_test)
    y_test = np.array(y_test, np.int32)
    np.savetxt("X_train", X_train)
    np.savetxt("y_train", y_train, fmt="%d")
    np.savetxt("X_test", X_test)
    np.savetxt("y_test", y_test, fmt="%d")
    X_train_load = np.loadtxt("X_train")
    y_train_load = np.loadtxt("y_train", dtype=np.int32)
    X_test_load = np.loadtxt("X_test")
    y_test_load = np.loadtxt("y_test", dtype=np.int32)
    print("save done")
    print(np.array_equal(X_train, X_train_load))
    print(np.array_equal(y_train, y_train_load))
    print(np.array_equal(X_test, X_test_load))
    print(np.array_equal(y_test, y_test_load))
    return X_train, y_train, X_test, y_test

def get_aloi_data_saved():
    X_train = np.loadtxt("X_train")
    y_train = np.loadtxt("y_train", dtype=np.int32)
    X_test = np.loadtxt("X_test")
    y_test = np.loadtxt("y_test", dtype=np.int32)
    return X_train, y_train, X_test, y_test

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %d/%d %s' % (prefix, bar, percent, iteration, total, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

# # make a list
# items = list(range(0, 57))
# i = 0
# l = len(items)
#
# # Initial call to print 0% progress
# printProgressBar(i, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
# for item in items:
#     # Do stuff...
#     sleep(0.1)
#     # Update Progress Bar
#     i += 1
#     printProgressBar(i, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
