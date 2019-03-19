# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:31:20 2019

@author: Charlie Chan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 21:17:51 2019

@author: Charlie Chan
"""

## 树叶种类识别
import os
import numpy as np
import pandas as pd

# 前期工作是分好训练集和测试集， 
# 990行数据， 99种花，每种花10个样本， train和test 8：2

os.chdir("./leaf-classification")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission = pd.read_csv("sample_submission.csv")

# all_type是所有的类别名字
all_type = list(submission.columns.values)[1:]

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
label = np.array(train['species'])
dic1 = {}
dic2 = {}
for i in enumerate(all_type):
    dic1[i[0]] = i[1]  # 数字 --> 类别
    dic2[i[1]] = i[0]  # 类别 --> 数字
res = []
for i in label:
    res.append([dic2[i]])
enc.fit(res)

from sklearn.cross_validation import train_test_split
X_train = pd.DataFrame()
X_test = pd.DataFrame()
y_train = []
y_test = []
for i in all_type:
    tmp_x = train.ix[list(np.where(label==i)[0]),:]
    tmp_y = label[list(np.where(label==i)[0])]
    a_train, a_test, b_train, b_test = train_test_split(tmp_x, tmp_y, test_size=0.2, random_state=42)
    X_train = pd.concat([X_train, a_train])
    X_test = pd.concat([X_test, a_test])
    y_train += list(b_train)
    y_test += list(b_test)

res1 = []
for i in y_train:
    res1.append([dic2[i]])
res2 = []
for i in y_test:
    res2.append([dic2[i]])

label_train = pd.DataFrame(enc.transform(res1).toarray())
label_test = pd.DataFrame(enc.transform(res2).toarray())

X_train = X_train.iloc[:, 2:]
X_test = X_test.iloc[:, 2:]



# tensorflow训练
import tensorflow as tf
import random as rdm
learning_rate = 0.05
x = tf.placeholder(tf.float32, [None, 192])
label = tf.placeholder(tf.float32, [None, 99])

# 隐藏层1： 120个神经元， ReLu
in_units = 192
h1_units = 120
h2_units = 120
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h2_units, 99]))
b2 = tf.Variable(tf.zeros([99]))

keep_prob = tf.placeholder(tf.float32)
hidden1 = tf.nn.relu(tf.matmul(x, W1)+b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)  #使用dropout抵抗过拟合
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2)+b2)

loss = -tf.reduce_sum(label * tf.log(y))
train = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

predict = tf.equal(tf.argmax(label, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(predict, "float"))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 100
train_step = 2000
for i in range(train_step):
    index = rdm.sample(range(792), batch_size)
    batch_x = X_train.iloc[index, :]
    batch_label = label_train.iloc[index, :]
    _, batch_loss = sess.run([train, loss], 
                             feed_dict={x: batch_x, label: batch_label, keep_prob:0.8})
    
    if (i + 1) % 100 == 0:
        print('第%5d步，当前loss：%.2f' % (i + 1, batch_loss))

accuracy_test = sess.run(accuracy,
                    feed_dict={x: X_test, label: label_test, keep_prob:1.0})
print("准确率: %.2f，共测试了%d张图片 " % (accuracy_test, len(label_test)))

##在测试集的准确率有96%