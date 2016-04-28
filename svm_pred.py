import numpy as np
import xlrd
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from math import ceil, floor

"""read data
"""
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


# Split the data for cross validation
X_train, X_test, Y_train, Y_test = train_test_split(preprocess_input, output_set,
                                                    test_size=0.1,
                                                    random_state=0)

"""helper function
"""
#accuracy
def accurayc(pred_labels, labels):
    correct = 0
    total = len(labels)
    for i in range(len(labels)):
        pred = pred_labels[i]
        label = labels[i]
        if pred == label:
            correct += 1
    return float(correct) / float(total)


"""Run svm
"""

lsvc = svm.LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, Y_train)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X_train)
clf = svm.SVC(kernel='rbf')
clf.fit(X_new, Y_train)
train_pred = clf.predict(X_new)
X_new = model.transform(X_test)
test_pred = clf.predict(X_new)
print "train accuracy: ", accurayc(train_pred, Y_train)
print "test accuracy: ", accurayc(test_pred, Y_test)