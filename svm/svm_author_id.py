#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
sys.path.append("../tools/")

from email_preprocess import preprocess



# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# For speeding up the classifier, by using smaller dataset
# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

svm_clf = svm.SVC(kernel='rbf', C=10000, gamma='auto')
t0 = time()
print('SVM Training begins...')
# Training the linear SVM classifier
svm_clf.fit(features_train, labels_train)
t_train = time() - t0
print('SVM Training ends...')
print("Training Time:", round(t_train, 3), "s")

t0 = time()
# Testing the SVM Model
print('SVM Testing begins...')
predicted_labels = svm_clf.predict(features_test)
t_test = time() - t0
print('SVM Testing ends...')
print("Testing Time:", round(t_test, 3), "s")

# Evaluating the model
accuracy = accuracy_score(y_true=labels_test, y_pred=predicted_labels)
hammingLoss = hamming_loss(y_true=labels_test, y_pred=predicted_labels)
print("Accuracy of Linear SVM Classifier is: ", accuracy)
print("Hamming Loss(Accuracy of Incorrect classification) of Linear SVM CLassifier is: ", hammingLoss)
