import numpy as np
import xlrd
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from math import ceil, floor
from random import shuffle

"""read data
"""
xl_wb = xlrd.open_workbook("set02.xlsx")

# Read the data
sheet_names = xl_wb.sheet_names()

input_set = [] # list of data
output_set = [] # 1 ~ 9
for i in range(1, len(sheet_names)):
    sheet_name = sheet_names[i]
    xl_ws = xl_wb.sheet_by_name(sheet_name)
    for j in range(xl_ws.ncols):
        instance = []
        if j % 2 == 0:
            continue
        column = xl_ws.col(j)
        for k in range(len(column)):
            if k <= 404:
                continue
            cell = column[k]
            value = cell.value
            instance.append(value)
        input_set.append(instance)
        if i >= 1 and i <= 6:
            output_set.append(0)
        elif i > 6 and i <= 9:
            output_set.append(1)
        else:
            output_set.append(2)

"""
# change the size of the feature set
tmp_input = list(input_set)
input_set = []
size = len(tmp_input[0]) / 2
for i in range(len(tmp_input)):
    input_feature = []
    for j in range(size):
        feature_value = (float(tmp_input[i][2*j]) + float(tmp_input[i][2*j+1])) / 2.0
        input_feature.append(feature_value)
    input_set.append(input_feature)
"""

# Handling continuous value
flatten_input = []
for instance in input_set:
    flatten_input += instance
max_value = max(flatten_input)
min_value = min(flatten_input)

gap = max_value - min_value
gap_fold_num = 20
gap_fold_size = float(gap) / float(gap_fold_num)
preprocess_input = []
for instance in input_set:
    preprocess_instance = []
    for value in instance:
        tmp_value = value - min_value
        value = float(floor(float(tmp_value) / gap_fold_size))
        preprocess_instance.append(value)
    preprocess_input.append(preprocess_instance)
"""
gap = max_value - min_value
preprocess_input = []
for instance in input_set:
    preprocess_instance = []
    for value in instance:
        tmp_value = value - min_value
        value = tmp_value / gap
        preprocess_instance.append(value)
    preprocess_input.append(preprocess_instance)
"""

# Split the data for cross validation
"""
X_train, X_test, Y_train, Y_test = train_test_split(preprocess_input, output_set,
                                                    test_size=0.1,
                                                    random_state=0)
"""








"""helper function
"""
#accuracy
def accuracy(pred_labels, labels):
    correct = 0
    total = len(labels)
    for i in range(len(labels)):
        pred = pred_labels[i]
        label = labels[i]
        if pred == label:
            correct += 1
    return float(correct) / float(total)

def split_dataset(index, fold_size, dataset):
    start_index = fold_size * index
    if index == 9:
        end_index = -1
    else:
        end_index = fold_size * (index + 1)
    train = []
    test = []
    test = dataset[start_index:end_index]
    train = dataset[:start_index] + dataset[end_index:-1]
    X_train = []
    Y_train = []
    for item in train:
        X_train.append(item[0])
        Y_train.append(item[1])
    X_test = []
    Y_test = []
    for item in test:
        X_test.append(item[0])
        Y_test.append(item[1])
    return (X_train, X_test, Y_train, Y_test)





"""Run svm

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, Y_train)
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)
print "train accuracy: ", accuracy(train_pred, Y_train)
print "test accuracy: ", accuracy(test_pred, Y_test)
# test accuracy = 67.95%
# train accuracy = 51.72%
"""

# Cross validation: 10 fold
size = len(preprocess_input)
dataset = [(preprocess_input[i], output_set[i]) for i in range(size)]
shuffle(dataset)
fold_size = size / 10
test_accuracy = []
train_accuracy = []
for i in range(10):
    (X_train, X_test, Y_train, Y_test) = split_dataset(i, fold_size, dataset)
    clf = svm.SVC(kernel='rbf')
    clf.fit(X_train, Y_train)
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    train_acc = accuracy(train_pred, Y_train)
    test_acc = accuracy(test_pred, Y_test)
    print "train accuracy: ", train_acc
    print "test accuracy: ", test_acc
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)

