import tensorflow as tf
sess = tf.Session()
from Utils.ops import *
import time
import numpy as np
from Utils import simple_env
'''
np.random.seed(111)
tf.set_random_seed(111)
'''
print('hi',sess.run(tf.random_uniform((1,))),np.random.rand())
''' hyper parameters'''
s_dim = 2
hid_dim = 64
z_dim = 2
lr = 1e-3
mb_dim = 320
mem_dim = 1000
'''setup graph'''
def make_network(inp,scope='network',reuse=False):
    #initial = tf.contrib.layers.xavier_initializer()
    initial = orthogonal_initializer()
    with tf.variable_scope(scope,reuse=reuse):
        hid = linear(inp,hid_dim,'hid1',tf.nn.relu,init=initial)
        hid = linear(hid,hid_dim,'hid2',tf.nn.relu,init=initial)
        last_hid = linear(hid,z_dim,'hid3',init=initial)
    return last_hid
_s = tf.placeholder(tf.float32,shape=(None,s_dim))
_r = tf.placeholder(tf.float32,shape=(None,1))
_mem_s = tf.placeholder(tf.float32,shape=(None,s_dim))
_mem_r = tf.placeholder(tf.float32,shape=(None,1))

mem_z = make_network(_mem_s)
z = make_network(_s,reuse=True)
sim = tf.nn.softmax(tf.reduce_sum(tf.expand_dims(z,1)*mem_z,-1))
pred_r = tf.matmul(sim,_mem_r)
loss = tf.reduce_mean(tf.square(pred_r-_r))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
env = simple_env.Simple(3)
S = np.zeros((mem_dim,s_dim))
R = np.zeros((mem_dim,1))
cum_loss = 0
s = np.zeros((mb_dim,s_dim))
r = np.zeros((mb_dim,1))
import matplotlib.pyplot as plt
plt.ion()
for i in range(int(1e4)):
    for j in range(mb_dim):
        s[j] = env.observation_space.sample()
        #r[j],_ = env.get_reward(env.encode(s[j]))
        _,r[j],_ = env.get_transition(s[j],0)
    _,cur_loss,pred = sess.run([train_step,loss,pred_r],feed_dict={_s:s,_r:r,_mem_s:S,_mem_r:R})
    cum_loss += cur_loss
    if i % int(1e2) == 0:
        for j in range(mem_dim):
            S[j] = env.observation_space.sample()
            #R[j],_ = env.get_reward(env.encode(S[j]))
            _,R[j],_ = env.get_transition(S[j],0)
        #print(R.sum())
        print(i,cum_loss)
        cum_loss = 0
        Xs = env.encode(s[:,0])
        Ys = env.encode(s[:,1])
        plt.clf()
        plt.scatter(Xs,Ys,s=100,c=pred)
        axes = plt.gca()
        axes.set_xlim([-env.limit,env.limit])
        axes.set_ylim([-env.limit,env.limit])
        plt.pause(.01)

