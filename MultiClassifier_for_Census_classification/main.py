
from optparse import OptionParser
import os
import sys
import copy
import numpy as np
import pandas as pd
import scipy as sp


def get_data():

    train_data = np.loadtxt('train_tensor.data', delimiter=',')
    test_data = np.loadtxt('test_tensor.data', delimiter=',')
    train_label = np.loadtxt('train_label.data')
    test_label = np.loadtxt('test_label.data')
    return train_data, train_label, test_data, test_label


def train(Model, model_name):

    train_data, train_label, test_data, test_label = get_data()
    model = Model
    model.fit(train_data, train_label)
    predict = model.predict(test_data)
    count = 0
    for left, right in zip(predict, test_label):
        if int(left) == int(right):
            count += 1
    print("{} accuracy : {}".format(model_name, float(count) / len(test_label)))


def get_model_from_name(model_name):

    if model_name == "MultinomialNB":
        from sklearn.naive_bayes import MultinomialNB
        return MultinomialNB(alpha=1.0), model_name
    elif model_name == "BernoulliNB":
        from sklearn.naive_bayes import BernoulliNB
        return BernoulliNB(alpha=0.1), model_name
    elif model_name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=2), model_name
    elif model_name == "DecisionTree":
        from sklearn import tree
        return tree.DecisionTreeClassifier(), model_name
    elif model_name == "LogisticRegression":
        from sklearn.linear_model.logistic import LogisticRegression
        return LogisticRegression(), model_name
    elif model_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_neighbors=2), model_name
    elif model_name == "NeuralNetwork":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(solver='adam', alpha=1e-2, hidden_layer_sizes=(100, 50), random_state=1), model_name
    else:
        raise ValueError("Not valid model")


def family():
    for MODELs in ["MultinomialNB", "BernoulliNB", "RandomForest", "DecisionTree", "LogisticRegression",
                   "KNN", "NeuralNetwork"]:
        Model, model_name = get_model_from_name(MODELs)
        train(Model, model_name)

family()
