from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
from sklearn import preprocessing
import pandas as pd
from numpy import dot
from numpy.linalg import norm


def euclidean_distance(test_row, row):  # euclidean_distance gives the best accuracy for our model
    distance = 0.0
    for i in range(len(test_row)):
        distance += (test_row[i] - row[i]) ** 2
    return sqrt(distance)


def jaccard_distance(x, y):  # jaccard_distance gives the second best accuracy for our model
    intersection = len(set(x).intersection(set(y)))
    union = len(set(x).union(set(y)))
    return 1 - intersection / union


def cosine_distance(x, y):  # cosine_distance gives the worst accuracy for our model
    return dot(x, y) / (norm(x) * norm(y))


def _train_test_split(x, y, test_size=0.3, random_state=41):  # split the data into testing and training parts
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def most_common(lst):  # find the most common element in a list
    temp = list()
    for label, distance in lst:
        temp.append(label[0])
    return max(set(temp), key=temp.count)


def knn_classifier(x_test, x_train, y_test, y_train):
    k = 3
    distances = list()  # stores the distance of x_test points to x_train points
    predicted_results = list()  # stores the predicted outcome

    x_test = list(x_test.values)
    x_train = list(x_train.values)
    y_test = list(y_test.values)
    y_train = list(y_train.values)

    for i in range(len(x_test)):  # iterate over the test data
        for j in range(len(x_train)):  # for each test data, iterate over the training data and calculate distance
            distances.append((y_train[j], euclidean_distance(x_test[i], x_train[j])))  # append the distance

        predicted_outcome = most_common(sorted(distances, key=lambda x: x[1])[:k])  # find the most common outcome
        predicted_results.append(predicted_outcome)  # append the predicted outcome
        distances = list()

    return predicted_results


def get_confusion_matrix(y_test, predicted):  # returns the confusion matrix of a model
    return confusion_matrix(y_true=y_test, y_pred=predicted)


def get_accuracy(y_test, predicted):  # returns the accuracy of a model
    return round(accuracy_score(y_test, predicted) * 100, 2)


def get_precision(y_test, predicted):
    return round(precision_score(y_test, predicted) * 100, 2)


def get_f1_score(y_test, predicted):
    return round(f1_score(y_test, predicted) * 100, 2)
