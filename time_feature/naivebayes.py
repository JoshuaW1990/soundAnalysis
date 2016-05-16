import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
import xlrd
from math import floor, ceil, log, exp
from collections import defaultdict
import matplotlib.pyplot as pl
from sklearn.feature_selection import SelectFromModel

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
41.38%
42.09%
"""

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





clf = MultinomialNB()
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_train)
correct = 0.0
for i in range(len(Y_train)):
    if Y_train[i] == Y_pred[i]:
        correct += 1.0
train_accuracy1 = correct / float(len(Y_train))
print train_accuracy1

Y_pred = clf.predict(X_test)
correct = 0.0
for i in range(len(Y_test)):
    if Y_test[i] == Y_pred[i]:
        correct += 1.0
test_accuracy1 = correct / float(len(Y_test))
print test_accuracy1







