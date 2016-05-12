import numpy as np
from sklearn import neighbors
from sklearn.cross_validation import train_test_split
import xlrd
from math import floor, ceil, log, exp
from collections import defaultdict
import matplotlib.pyplot as plt


xl_wb = xlrd.open_workbook("set01.xlsx")

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
        output_set.append(i)

# Handling continuous value
flatten_input = []
for instance in input_set:
    flatten_input += instance
max_value = max(flatten_input)
min_value = min(flatten_input)
gap = max_value - min_value
gap_fold_num = 10
gap_fold_size = float(gap) / float(gap_fold_num)
preprocess_input = []
for instance in input_set:
    preprocess_instance = []
    for value in instance:
        tmp_value = value - min_value
        value = int(floor(float(tmp_value) / gap_fold_size))
        preprocess_instance.append(value)
    preprocess_input.append(preprocess_instance)





# Split the data for cross validation
X_train, X_test, Y_train, Y_test = train_test_split(preprocess_input, output_set,
                                                    test_size=0.1,
                                                    random_state=0)
"""KNN method
"""
n_neighbors = 50
clf = neighbors.KNeighborsClassifier(10, weights='uniform')
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
correct = 0.0
for i in range(len(Y_test)):
    if Y_test[i] == Y_pred[i]:
        correct += 1.0
test_accuracy1 = correct / float(len(Y_test))
print "test accuracy: ", test_accuracy1

Y_pred = clf.predict(X_train)
correct = 0.0
for i in range(len(Y_train)):
    if Y_train[i] == Y_pred[i]:
        correct += 1.0
train_accuracy1 = correct / float(len(Y_train))
print "train accuracy: ", train_accuracy1


