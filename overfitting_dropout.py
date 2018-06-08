#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import os 
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


#load data

digits=load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)

# add layer function
def add_layer(inputs, in_size,out_size,n_layer,activation_function=None):
    #add one more layer and return the ouput of this layer
    layer_name = n_layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name="W")
            tf.summary.histogram(str(layer_name)+"/weights",Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
            tf.summary.histogram(str(layer_name)+"/biases",biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights) + biases
            Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(str(layer_name)+"/outputs",outputs)
        return outputs

#define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,64],name='x_input')
    ys = tf.placeholder(tf.float32,[None,10],name='y_input')
    keep_prob = tf.placeholder(tf.float32)


#add output layer
l1 = add_layer(xs, 64, 100,'l1' ,activation_function=tf.nn.tanh)
prediction = add_layer(l1,100,10,'l2',activation_function=tf.nn.softmax)


#the loss between prediction and real data
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(tf.clip_by_value(prediction,1e-8,1.0)),reduction_indices=[1]))
    tf.summary.scalar('loss',cross_entropy)


with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)


sess = tf.Session()

#define mergeed
merged = tf.summary.merge_all()

#summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train",sess.graph)
test_writer = tf.summary.FileWriter("logs/test",sess.graph)


#writer = tf.summary.FileWriter("logs/",sess.graph)

init = tf.global_variables_initializer() #replace with global_variables_initializer
sess.run(init)



for i in range(10000):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.5})
    #sess.run(train_step,feed_dict={xs:X_train,ys:y_train})
    if i%50==0:
	#pass
        #recorde loss
        train_result=sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
        test_result=sess.run(merged,feed_dict={xs:X_test,ys:y_test,keep_prob:1})
        #train_result=sess.run(merged,feed_dict={xs:X_train,ys:y_train})
        #test_result=sess.run(merged,feed_dict={xs:X_test,ys:y_test})
	train_writer.add_summary(train_result,i)
	test_writer.add_summary(test_result,i)
