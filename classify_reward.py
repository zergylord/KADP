import tensorflow as tf
sess = tf.Session()
import time
cur_time = time.clock()
import numpy as np
from matplotlib import pyplot as plt
from Utils.ops import linear
plt.ion()    
eps = 1e-10
'''
def bce(o,t):
    return -tf.reduce_mean(tf.log(o)*t+tf.log(1-o)*(1-t))
'''
def bce(o,t):
    return -tf.reduce_mean(tf.log(tf.clip_by_value(o,eps,np.Inf))*t+tf.log(tf.clip_by_value(1-o,eps,np.Inf))*(1-t))
def mse(o,t):
    return tf.reduce_mean(tf.square(t-o))
'''
def linear(in_,out_dim,name,activation_fn=None,tied=False):
    in_dim = in_.get_shape()[1]
    with tf.variable_scope(name,reuse=tied):
        W = tf.get_variable('W',[in_dim,out_dim],tf.float32)
        b = tf.get_variable('b',[out_dim],tf.float32)
        out = tf.matmul(in_,W) + b
        if activation_fn != None:
            mean,variance = tf.nn.moments(out,[0,1])
            beta = tf.Variable(tf.constant(0.0,shape=[out_dim]))
            gamma = tf.Variable(tf.constant(1.0,shape=[out_dim]))
            out = tf.nn.batch_normalization(out,mean,variance,beta,gamma,eps)
            out = activation_fn(out)
    return out
'''

mb_dim = 32
hid_dim = 1000 
lr = 4e-5
region = 1
refresh = int(1e3)
def get_data(mb_dim):
    xmin = -.1
    ymin = -.2
    X = 2*(np.random.rand(mb_dim,2)-.5)
    Y = np.zeros((mb_dim,1))
    #Y[(X[:,0]>xmin)*(X[:,0]<xmin+region),0] = 1
    Y[
            (X[:,0] > xmin) *
            (X[:,0] < xmin+region) *
            (X[:,1] > ymin) *
            (X[:,1] < ymin+region)
            ] = 1.0
    return X,Y
'''
def get_data(mb_dim):
    X = np.random.randn(mb_dim,1)
    Y = X**2
    return X,Y
'''
inp = tf.placeholder(tf.float32,shape=(None,2))
target = tf.placeholder(tf.float32,shape=(None,1))
last_hid = linear(inp,hid_dim,'hid1',tf.nn.tanh)
last_hid = linear(last_hid,hid_dim,'hid2',tf.nn.tanh)
output = linear(last_hid,1,'reward',tf.nn.sigmoid)
loss = bce(output,target)
'''
output = linear(last_hid,1,'reward')
loss = mse(output,target)
'''
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

sess.run(tf.global_variables_initializer())

cum_loss = 0.0
loss_hist = []
for i in range(int(1e5)):
    X,Y = get_data(mb_dim)
    _,cur_loss,O = sess.run([train_step,loss,output],feed_dict={inp:X,target:Y})
    cum_loss+=cur_loss
    if i % refresh == (refresh-1):
        print(i,Y.sum()/mb_dim,O.sum()/mb_dim,cum_loss/mb_dim,time.clock()-cur_time)
        cur_time = time.clock()
        loss_hist.append(cum_loss/mb_dim)
        cum_loss = 0.0
        plt.figure(0)
        plt.clf()
        plt.plot(loss_hist)
        plt.pause(.1)

        plt.figure(1)
        plt.clf()
        plt.subplot(1,2,1)
        plt.scatter(X[:,0],X[:,1],s=100,c=O)
        #plt.scatter(X[:,0],O)
        plt.subplot(1,2,2)
        plt.scatter(X[:,0],X[:,1],s=100,c=Y)
        #plt.scatter(X[:,0],Y)
    
