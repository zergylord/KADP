import numpy as np
import tensorflow as tf
from ops import *
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
in_dim = 28*28
hid_dim = 128
k = 64
out_dim = 10
lr = 1e-3
eps = 1e-10
rho = .999
x_ = tf.placeholder(tf.float32,shape=[1,in_dim])
y_ = tf.placeholder(tf.float32,shape=[1,out_dim])
hid = linear(x_,hid_dim,'hid')
hid_k,ind_k = tf.nn.top_k(hid,k,sorted=False)
with tf.variable_scope('local_winner'):
    W = tf.get_variable('W',[hid_dim,out_dim],tf.float32,initializer=orthogonal_initializer())
    b = tf.get_variable('b',[out_dim],tf.float32)
label_prob = tf.nn.softmax(tf.matmul(tf.expand_dims(tf.gather(hid[0],ind_k[0]),0),tf.gather(W,ind_k[0])) + b)
#label_prob = linear(hid_k,out_dim,'pooling',tf.nn.softmax)
#label_prob = linear(tf.nn.relu(hid),out_dim,'normal',tf.nn.softmax)
acc = tf.reduce_mean(tf.to_float(tf.nn.in_top_k(label_prob,tf.arg_max(y_,1),1)))
loss = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(label_prob,eps,1)))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
cumloss = 0
cumacc = .1
refresh = int(1e4)
for i in range(int(1e6)):
    mb_input,mb_target = mnist.train.next_batch(1)
    _,mb_loss,mb_acc = sess.run([train_step,loss,acc],feed_dict={x_:mb_input,y_:mb_target})
    cumloss = rho*cumloss + (1-rho)*mb_loss
    cumacc = rho*cumacc + (1-rho)*mb_acc
    if i % refresh == 0:
        print(i,cumloss,cumacc)
