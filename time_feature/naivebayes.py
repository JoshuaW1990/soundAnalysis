import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
import xlrd
from math import floor, ceil, log, exp
from collections import defaultdict
import matplotlib.pyplot as pl
from sklearn.feature_selection import SelectFromModel

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


"""predict with naive bayes model with bigram

# model with sequenctial charactersitics
class BigramNB:
    log_prob = defaultdict(defaultdict) # conditional probability: p(word|c) = (count(c, word) + 1) / (count(c) + |v|)
    vocabulary = set()
    labels = set()
    bigram_count = defaultdict(defaultdict)
    unigram_count = defaultdict(defaultdict) # count(c) key: label, value:  number of words

    def __init__(self, input = None, output = None):
        if input != None:
            for instance in input:
                for item in instance:
                    self.vocabulary.add(item)
        if output != None:
            self.labels = set(output)
        if input != None and output != None:
            self.train(input, output)

    def train(self, input, output):
        # Count the frequency of the label and word
        for i in range(len(input)):
            label = output[i]
            instance = input[i]
            if label not in self.unigram_count:
                self.unigram_count[label] = defaultdict(float)
            unigram_dict = self.unigram_count[label]
            if label not in self.bigram_count:
                self.bigram_count[label] = defaultdict(float)
            bigram_dict = self.bigram_count[label]
            for j in range(1, len(instance)):
                unigram = instance[j]
                bigram = (instance[j - 1], instance[j])
                unigram_dict[unigram] += 1.0
                bigram_dict[bigram] += 1.0

        # Calculate the logaritmic value of probability
        for label in self.labels:
            if label not in self.log_prob:
                self.log_prob[label] = defaultdict(float)
            tmp_log_prob = self.log_prob[label]
            for bigram in self.bigram_count[label].keys():
                unigram = bigram[0]
                tmp_log_prob[bigram] = log(float(self.bigram_count[label][bigram] + 1) / float(self.unigram_count[label][unigram] + len(self.vocabulary)))
        return


    def classify(self, instance):
        max_prob = 0.0
        pred_label = list(self.labels)[0]
        for label in self.labels:
            pred_prob = 0.0
            for i in range(1, len(instance)):
                bigram = (instance[i - 1], instance[i])
                if bigram not in self.log_prob[label]:
                    print "error"
                    return None
                else:
                    tmp_prob = self.log_prob[label][bigram]
                pred_prob += tmp_prob
            pred_prob = exp(pred_prob)
            if pred_prob > max_prob:
                max_prob = pred_prob
                pred_label = label
        return pred_label

    def predict(self, input):
        pred_output = []
        for instance in input:
            pred_label = self.classify(instance)
            pred_output.append(pred_label)
        return pred_output



model = BigramNB(X_train, Y_train)
test_pred = model.predict(X_test)
correct = 0.0
for i in range(len(Y_test)):
    if Y_test[i] == test_pred[i]:
        correct += 1.0
test_accuracy1 = correct / float(len(Y_test))
print test_accuracy1
train_pred = model.predict(X_train)
correct = 0.0
for i in range(len(Y_train)):
    if Y_train[i] == train_pred[i]:
        correct += 1.0
train_accuracy1 = correct / float(len(Y_train))
print train_accuracy1
"""

# Not good
lsvc = MultinomialNB().fit(X_train, Y_train)
model = SelectFromModel(lsvc, prefit=True)
X_train = model.transform(X_train)
X_test = model.transform(X_test)


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




"""Neural Network

x_dimension = len(X_train[0])
y_dimension = 1
ds = SupervisedDataSet(x_dimension, y_dimension)
for i in range(len(X_train)):
    ds.addSample(X_train[i], Y_train[i])

net = buildNetwork(x_dimension, x_dimension, y_dimension, bias=True)

trainer = BackpropTrainer(net, dataset = ds)
trnerr,valerr = trainer.trainUntilConvergence(dataset=ds, maxEpochs=50)
pl.plot(trnerr,'b',valerr,'r')

test_ds = SupervisedDataSet(x_dimension, y_dimension)
for i in range(len(X_test)):
    test_ds.addSample(X_test[i], Y_test[i])
train_out = net.activateOnDataset(ds)
Y_pred = [int(item) for item in train_out]
correct = 0.0
for i in range(len(Y_train)):
    if Y_train[i] == Y_pred[i]:
        correct += 1.0
train_accuracy2 = correct / float(len(Y_train))
print train_accuracy2
test_out = net.activateOnDataset(test_ds)
Y_pred = [int(item) for item in test_out]
correct = 0.0
for i in range(len(Y_test)):
    if Y_test[i] == Y_pred[i]:
        correct += 1.0
test_accuracy2 = correct / float(len(Y_test))
print test_accuracy2
"""


"""svm
"""

