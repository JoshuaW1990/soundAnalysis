import numpy as np
import xlrd
import csv
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from math import floor, ceil, log, exp
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
            if k <= 1 or k >= 402:
                continue
            cell = column[k]
            value = cell.value
            instance.append(value)
        input_set.append(instance)
        output_set.append(i)


def find_peak(wave_data):
    poss_max = []
    poss_min = []
    for i in range(1, len(wave_data) - 1):
        prev_delta = wave_data[i] - wave_data[i - 1]
        post_delta = wave_data[i + 1] - wave_data[i]
        if prev_delta < 0 and post_delta > 0:
            poss_min.append(i)
        if prev_delta > 0 and post_delta < 0:
            poss_max.append(i)
    return (poss_min, poss_max)


def get_feature(wave_data):
    (poss_min, poss_max) = find_peak(wave_data)
    amplitude = []
    frequency = []
    for i in range(1, len(poss_max)):
        if i > 10:
            continue
        max_value = wave_data[poss_max[i]]
        min_value = wave_data[poss_min[i - 1]]
        amplitude.append(max_value - min_value)
        frequency.append(poss_max[i] - poss_max[i - 1])
    """
    if i < 10:
        make_up = [ None for j in range(10 - i)]
        amplitude = amplitude + make_up
        frequency = frequency + make_up
    """
    return (amplitude, frequency)


def obtain_features(input, output):
    amplitude_list = []
    frequency_list = []
    preprocessed_output = []
    count = 0
    for idx, instance in enumerate(input):
        (amplitude, frequency) = get_feature(instance)
        if len(amplitude) < 10:
            print "error"
            count += 1
            continue
        amplitude_list.append(amplitude)
        frequency_list.append(frequency)
        preprocessed_output.append(output[idx])
    print "total error: ", count
    # Use the average value to replace the None place
    for i in range(10):
        total_amplitude = 0
        total_frequency = 0
        count = len(amplitude_list)
        for j in range(len(amplitude_list)):
            amplitudes = amplitude_list[j]
            frequencies = frequency_list[j]
            if amplitudes[i] != None:
                total_amplitude += amplitudes[i]
            if frequencies[i] != None:
                total_frequency += frequencies[i]
        average_amplitude = float(total_amplitude) / float(count)
        average_frequency = float(total_frequency) / float(count)
        for j in range(len(amplitude_list)):
            if amplitude_list[j][i] == None:
                amplitude_list[j][i] = average_amplitude
            if frequency_list[j][i] == None:
                frequency_list[j][i] = average_frequency
    # Normalization
    flatten_frequency = []
    for frequencies in frequency_list:
        flatten_frequency += frequencies
    max_frequency = max(flatten_frequency)
    for i in range(len(amplitude_list)):
        amplitudes = amplitude_list[i]
        frequencies = frequency_list[i]
        max_amplitude = max(amplitudes)
        for j in range(len(amplitudes)):
            amplitudes[j] = float(amplitudes[j]) / float(max_amplitude)
            frequencies[j] = float(frequencies[j]) / float(max_frequency)

    # Discreticize the data

    flatten_list = []
    for item in amplitude_list:
        flatten_list += item
    max_value = max(flatten_list)
    min_value = min(flatten_list)
    gap = max_value - min_value
    gap_fold_num = 20
    gap_fold_size = float(gap) / float(gap_fold_num)
    preprocess_amplitude = []
    for instance in amplitude_list:
        preprocess_instance = []
        for value in instance:
            tmp_value = value - min_value
            value = int(floor(float(tmp_value) / gap_fold_size))
            preprocess_instance.append(value)
        preprocess_amplitude.append(preprocess_instance)

    flatten_list = []
    for item in frequency_list:
        flatten_list += item
    max_value = max(flatten_list)
    min_value = min(flatten_list)
    gap = max_value - min_value
    gap_fold_num = 20
    gap_fold_size = float(gap) / float(gap_fold_num)
    preprocess_freqency = []
    for instance in frequency_list:
        preprocess_instance = []
        for value in instance:
            tmp_value = value - min_value
            value = int(floor(float(tmp_value) / gap_fold_size))
            preprocess_instance.append(value)
        preprocess_freqency.append(preprocess_instance)
    return (preprocess_amplitude, preprocess_freqency, preprocessed_output)

def merge_features(input, output):
    (amplitude_list, frequency_list, preprocessed_output) = obtain_features(input, output)
    preprocessed_input = []
    for i in range(len(amplitude_list)):
        tmp_features = amplitude_list[i] + frequency_list[i]
        preprocessed_input.append(tmp_features)
    return (preprocessed_input, preprocessed_output)


(preprocessed_input, preprocessed_output) = merge_features(input_set, output_set)

# Split the data for cross validation
X_train, X_test, Y_train, Y_test = train_test_split(preprocessed_input, preprocessed_output,
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
"""
lsvc = svm.LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, Y_train)
model = SelectFromModel(lsvc, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)
# test accuracy = 37.93%
# train accuracy = 62.93%
"""
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, Y_train)
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)
print "train accuracy: ", accurayc(train_pred, Y_train)
print "test accuracy: ", accurayc(test_pred, Y_test)
# test accuracy = 67.95%
# train accuracy = 51.72%



"""naive bayes model
"""
clf = MultinomialNB()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
correct = 0.0
for i in range(len(Y_test)):
    if Y_test[i] == Y_pred[i]:
        correct += 1.0
test_accuracy1 = correct / float(len(Y_test))
print test_accuracy1
# raw: 37.9%
# preprocessed: 58.62%
Y_pred = clf.predict(X_train)
correct = 0.0
for i in range(len(Y_train)):
    if Y_train[i] == Y_pred[i]:
        correct += 1.0
train_accuracy1 = correct / float(len(Y_train))
print train_accuracy1

# accuracy = 0.6770833333333334
# preprocessed: 57.92%
