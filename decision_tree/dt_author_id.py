#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn import tree
from sklearn.metrics import accuracy_score, hamming_loss
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
dt_clf = tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()
print('Decision Tree Training begins...')
dt_clf.fit(features_train, labels_train)
t_train = time() - t0
print('Decision Tree Training ends...')
print("Training Time:", round(t_train, 3), "s")

t0 = time()
# Testing the Decision Tree Model
print('Decision Tree Testing begins...')
predicted_labels = dt_clf.predict(features_test)
t_test = time() - t0
print('Decision Tree Testing ends...')
print("Testing Time:", round(t_test, 3), "s")

# Evaluating the model
accuracy = accuracy_score(y_true=labels_test, y_pred=predicted_labels)
hamming_loss = hamming_loss(y_true=labels_test, y_pred=predicted_labels)
print("Accuracy of Decision Tree Classifier is: ", accuracy)
print("Hamming Loss(Accuracy of Incorrect classification) of Linear Decision Tree Classifier is: ", hamming_loss)



