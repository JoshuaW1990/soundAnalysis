import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import xlrd
from math import floor, ceil, log, exp
import matplotlib.pyplot as pl
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules   import LSTMLayer, SoftmaxLayer


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
        tag = [0. for idx in range(len(sheet_names) - 1)]
        tag[i - 1] = 1.0
        output_set.append(tag)

# Normalization
flatten_input = []
for instance in input_set:
    flatten_input += instance
max_value = max(flatten_input)
min_value = min(flatten_input)
gap = max_value - min_value
preprocess_input = []
for instance in input_set:
    preprocess_instance = []
    for value in instance:
        tmp_value = value - min_value
        value = tmp_value / gap
        preprocess_instance.append(value)
    preprocess_input.append(preprocess_instance)





# Split the data for cross validation
X_train, X_test, Y_train, Y_test = train_test_split(preprocess_input, output_set,
                                                    test_size=0.1,
                                                    random_state=0)

"""simple model

x = tf.placeholder("float", [None, 200])
W = tf.Variable(tf.zeros([200, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

y_ = tf.placeholder("float", [None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    if i % 100 == 0:
        print i
    sess.run(train_step, feed_dict={x: X_train, y_: Y_train})



correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
acc = sess.run(accuracy, feed_dict={x: X_test, y_: Y_test})

sess.close()

print acc
"""

def convert_output(pred_labels):
    result = []
    for label in pred_labels:
        tmp_label = list(label)
        index = tmp_label.index(max(tmp_label))
        new_label = [0.0 for i in range(len(label))]
        new_label[index] = 1.0
        result.append(new_label)
    return np.array(result)

def compare_list(list1, list2):
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            continue
        else:
            return False
    return True




x_dimension = len(X_train[0])
y_dimension = len(Y_train[0])
ds = SupervisedDataSet(x_dimension, y_dimension)
for i in range(len(X_train)):
    ds.addSample(X_train[i], Y_train[i])

net = buildNetwork(x_dimension, x_dimension, y_dimension, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, bias=True)

trainer = BackpropTrainer(net, dataset = ds)
trnerr,valerr = trainer.trainUntilConvergence(dataset=ds, maxEpochs=5000)
pl.plot(trnerr,'b',valerr,'r')

test_ds = SupervisedDataSet(x_dimension, y_dimension)
for i in range(len(X_test)):
    test_ds.addSample(X_test[i], Y_test[i])
train_out = net.activateOnDataset(ds)
Y_pred = convert_output(train_out)
correct = 0.0
for i in range(len(Y_train)):
    if compare_list(Y_train[i], Y_pred[i]):
        correct += 1.0
train_accuracy2 = correct / float(len(Y_train))
print "training accuracy is ", train_accuracy2
test_out = net.activateOnDataset(test_ds)
Y_pred = convert_output(test_out)
correct = 0.0
for i in range(len(Y_test)):
    if compare_list(Y_test[i], Y_pred[i]):
        correct += 1.0
test_accuracy2 = correct / float(len(Y_test))
print "test accuracy is ", test_accuracy2


"""
not lstm
epochs = 50
In[69]: run ann.py
0.150579150579
0.241379310345

epochs = 500
In[71]: run ann.py
0.420849420849
0.413793103448

epochs = 5000
In[73]: run ann.py
0.513513513514
0.448275862069


lstm
epochs = 50
training accuracy is  0.158301158301
test accuracy is  0.206896551724

epochs = 500
training accuracy is  0.11583011583
test accuracy is  0.172413793103

"""