import json
import os

def load_config(filename):
    """
    load config json file and read parameters

    parameters:
    filename: config json filename

    returns:
    train_file: training data filename
    test_file: testing data filename
    num_classes: number of labels in training file
    num_features: number of features in testing file
    """
    with open(filename) as f:
        config = json.load(f)
        train_file = config['train_file']
        test_file = config['test_file']
        num_classes = config['num_classes']
        num_features = config['num_features']
        return train_file, test_file, num_classes, num_features
