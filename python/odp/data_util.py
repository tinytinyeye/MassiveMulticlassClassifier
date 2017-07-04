from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from time import sleep
import numpy as np
from pathlib import Path
import os
import gzip
import gc
from hash import *

def get_odp_meta(filename):
    num_points = 0
    label_dict = {}
    max_feature = 0
    with gzip.open(filename, 'r') as f:
        lines = [x.decode('utf8').strip() for x in f.readlines()]
        for line in lines:
            num_points += 1
            if num_points % 10000 == 0:
                print("read {} lines".format(num_points))
            data = line.split('|')
            label = data[0].strip()
            if (label not in label_dict):
                label_dict[label] = 1
            features_str = data[1].strip()
            features = features_str.split(' ')
            for i in range(len(features)):
                feature = features[i].split(':')[0]
                if int(feature) > max_feature:
                    max_feature = int(feature)
    num_labels = len(label_dict.keys())
    num_features = max_feature
    return num_points, num_features, num_labels

def load_odp_data(filename, num_features):
    with gzip.open(filename, 'r') as f:
        # build a mapping from label to a list of data
        lines = [x.decode('utf8').strip() for x in f.readlines()]
        num_points = 0
        total = len(lines)
        print("finished loading lines")
        print("splitting data into labels")
        printProgressBar(num_points, total, prefix = 'Progress:', suffix = 'Complete', length = 50)
        label_to_data = {}
        for line in lines:
            num_points += 1
            printProgressBar(num_points, total, prefix = 'Progress:', suffix = 'Complete', length = 50)
            x_row = np.zeros(num_features)
            data = line.split('|')
            label = data[0].strip()
            features_str = data[1].strip()
            features = features_str.split(' ')
            for i in range(len(features)):
                feature = features[i].split(':')[0]
                x_row[int(feature[0])] = float(feature[1])
            if label not in label_to_data:
                label_to_data[label] = []
            label_to_data[label].append(x_row)
        print("finished")
        return label_to_data

def load_odp_data_raw(filename, num_features):
    X = []
    y = []
    with gzip.open(filename, 'r') as f:
        # build a mapping from label to a list of data
        lines = [x.decode('utf8').strip() for x in f.readlines()]
        num_points = 0
        total = len(lines)
        print("finished loading lines")
        printProgressBar(num_points, total, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for line in lines:
            x_row = np.zeros(num_features)
            data = line.split('|')
            label = data[0].strip()
            features_str = data[1].strip()
            features = features_str.split(' ')
            for i in range(len(features)):
                feature = features[i].split(':')[0]
                x_row[int(feature[0])] = float(feature[1])
            X.append(x_row)
            y.append(label)
            # Update Progress Bar
            num_points += 1
            # print("%d / %d completed" % (num_points, total))
            printProgressBar(num_points, total, prefix = 'Progress:', suffix = 'Complete', length = 50)
        X = np.array(X)
        y = np.array(y, np.int32)
        print("load finished")
        return X, y

def get_odp_data(filename, num_features, sample_size=1):
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
    X = []
    y = []
    num_labels = 0
    total = len(label_to_data.keys())
    print("start getting data from label, total", total)
    printProgressBar(num_labels, total, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for label in label_to_data:
        num_labels += 1
        printProgressBar(num_labels, total, prefix = 'Progress:', suffix = 'Complete', length = 50)
        data = label_to_data[label]
        # first shuffle data for a label
        data = shuffle(data)
        # trunck sample size
        data = data[:int(len(data) * sample_size)]
        X.extend(data)
        y.extend([label]*len(data))
    # shuffle data before
    X, y = shuffle(X, y)
    gc.collect()
    X = np.array(X)
    y = np.array(y, np.int32)
    print("getting data finished")
    return X, y

def save_odp_data_batch(load_filename, save_filename, num_features, n_batch=100):
    label_to_data = load_odp_data(load_filename, num_features)
    for b in range(0, n_batch + 1):
        printProgressBar(b, n_batch, prefix = 'Progress:', suffix = 'Complete', length = 50)
        X_file = Path("./data/batch/" + save_filename + "/X_" + save_filename + "_" + str(b) + ".gz")
        y_file = Path("./data/batch/" + save_filename + "/y_" + save_filename + "_" + str(b) + ".gz")

        # Check if the file is already saved
        if X_file.is_file() and y_file.is_file():
            continue
        else:
            X = []
            y = []
            for label in label_to_data:
                data = label_to_data[label]
                batch_size = int(len(data) / n_batch)
                if (b < n_batch):
                    data = data[b * batch_size : (b + 1) * batch_size]
                else:
                    data = data[b * batch_size:] # save the last batch
                X.extend(data)
                y.extend([label]*len(data))
            X = np.array(X)
            y = np.array(y, np.int32)
            np.savetxt("./data/batch/" + save_filename + "/X_" + save_filename + "_" + str(b) + ".gz", X)
            np.savetxt("./data/batch/" + save_filename + "/y_" + save_filename + "_" + str(b) + ".gz", y, fmt="%d")
    print()
    print("save all done")

def save_odp_data(filename, X, y):
    print("saving data")
    np.savetxt("X_" + filename + ".gz", X)
    np.savetxt("y_" + filename + ".gz", y, fmt="%d")
    # X_load = np.loadtxt("X_" + filename + ".gz")
    # y_load = np.loadtxt("y_" + filename + ".gz", dtype=np.int32)
    print("save done")
    # print(np.array_equal(X, X_load))
    # print(np.array_equal(y, y_load))

def get_odp_train_data(num_features, sample_size=1):
    return get_odp_data("./data/odp_train.vw.gz", num_features, sample_size)

def get_odp_test_data(num_features, sample_size=1):
    return get_odp_data("./data/odp_test.vw.gz", num_features, sample_size)

def get_odp_data_saved():
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

#
# Sample Usage
#



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
