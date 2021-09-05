#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


##############################################################
from sklearn.naive_bayes import GaussianNB
t0= time()
naive_bayes_clf = GaussianNB()
#Fitting the gaussian naive bayes classifier
naive_bayes_clf.fit(features_train, labels_train)
t_train = time() - t0
print("Training Time:", round(t_train, 3), "s")

t0= time()
predicted_labels = naive_bayes_clf.predict(features_test)
t_test = time() - t0
print("Testing Time:", round(t_test, 3), "s")

from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
accuracy = accuracy_score(y_true=labels_test, y_pred= predicted_labels)
hammingLoss = hamming_loss(y_true=labels_test, y_pred= predicted_labels)
print("Accuracy of Naive Bayes CLassifier is: ", accuracy)
print("Hamming Loss(Accuracy of Incorrect classification) of Naive Bayes CLassifier is: ", hammingLoss)


##############################################################

##############################################################
'''
You Will be Required to record time for Training and Predicting 
The Code Given on Udacity Website is in Python-2
The Following Code is Python-3 version of the same code
'''

# t0 = time()
# # < your clf.fit() line of code >
# print("Training Time:", round(time()-t0, 3), "s")

# t0 = time()
# # < your clf.predict() line of code >
# print("Predicting Time:", round(time()-t0, 3), "s")

##############################################################