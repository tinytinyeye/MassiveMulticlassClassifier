# MassiveMulticlassClassifier
### Introduction
Massive Multiclass Classifier is a demo to a new classification method that can handle classification task with large number of classes. 
To explain the idea of this method, we first define **B** as the number of merged classes and **R** as the number of sub-classifiers. Sub-classifier is just a classifier based on logistic regression.
For example, let B = 100 and R = 50, and our target dataset has 10k classes. At the beginning, we will first initiate 50 2-wise independent hash functions to convert labels from [0, 10000) to [0, 100). With each of these hash functions, the original 10000 classes will be merged to the 100 new classes randomly. These hash functions will be saved for training and predicting.
We will then train 50 sub-classifiers with logistic regression. Each sub-classifier will use one of the hash function we generated before to convert labels from training data, and utilize these new labels in training.
After training, for each sub-classifier, we will feed in prediction set to compute the probability for each merged class. To compute the probability for each original class, we do the following:
1. For each original class, we use the same set of hash functions to compute the corresponding merged class in each sub-classifier. 
2. Once we have the probability for the merged class of a original class in each sub-classifier, we will compute the average and use it as the score for that original class.
3. Then among all original classes, we will pick the one with highest score and choose it as the prediction of a data entry.
### Notes
Since each dataset has different process procedures and parameters, folders such as "odp" and "ltcb" represent the demo of that dataset.
* hash_util.py contains functions related to random hashing
* sparse_multiclf.py or dense_multiclf.py contains code for classifier.
* *_demo.py contains the main program to run the classifier.