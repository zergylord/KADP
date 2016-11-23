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
def make_encoder(inp,scope='encoder',reuse=False):
    #initial = tf.contrib.layers.xavier_initializer()
    initial = orthogonal_initializer()
    with tf.variable_scope(scope,reuse=reuse):
        hid = linear(inp,hid_dim,'hid1',tf.nn.relu,init=initial)
        hid = linear(hid,hid_dim,'hid2',tf.nn.relu,init=initial)
        last_hid = linear(hid,z_dim,'hid3',init=initial)
    return last_hid
def kernel(z,mem_z,mother='dot'):
    if mother == 'rbf':
        b = 1
        rbf = tf.exp(-tf.reduce_sum(tf.square(tf.expand_dims(z,1)-mem_z),-1)/b) 
        normed = rbf/tf.reduce_sum(rbf,-1,keep_dims=True)
        return normed,rbf
    elif mother == 'dot':
        dot = tf.reduce_sum(tf.expand_dims(z,1)*mem_z,-1)
        return tf.nn.softmax(dot),dot
    else:
        print('nope')
def kl(p,q):
    return tf.reduce_mean(tf.reduce_sum(p*tf.log(p/q),-1))
def mse(o,t):
    return tf.reduce_mean(tf.reduce_sum(tf.square(o-t),-1))
_s = tf.placeholder(tf.float32,shape=(None,s_dim))
_r = tf.placeholder(tf.float32,shape=(None,1))
_rP = tf.placeholder(tf.float32,shape=(None,1))
_rPP = tf.placeholder(tf.float32,shape=(None,1))
_sPrime = tf.placeholder(tf.float32,shape=(None,s_dim))
_mem_s = tf.placeholder(tf.float32,shape=(None,s_dim))
_mem_r = tf.placeholder(tf.float32,shape=(None,1))
_mem_sPrime = tf.placeholder(tf.float32,shape=(None,s_dim))
'''embedings'''
mem_z = make_encoder(_mem_s)
mem_zPrime = make_encoder(_mem_sPrime,reuse=True)
z = make_encoder(_s,reuse=True)
zPrime = make_encoder(_sPrime,reuse=True)
'''similarity'''
sim,_ = kernel(z,mem_z)
mem_sim,_ = kernel(mem_zPrime,mem_z)
pred_r = tf.matmul(sim,_mem_r),
pred_rP = tf.matmul(tf.matmul(sim,mem_sim),_mem_r)
pred_rPP = tf.matmul(tf.matmul(tf.matmul(sim,mem_sim),mem_sim),_mem_r)
r_loss = mse(pred_r,_r)
rP_loss = mse(pred_rP,_rP)
rPP_loss = mse(pred_rPP,_rPP)
'''loss'''
loss = r_loss + rP_loss + rPP_loss
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
#train_step = tf.no_op()

#check_op = tf.add_check_numerics_ops()
check_op = tf.no_op()
sess = tf.Session()
sess.run(tf.initialize_all_variables())
env = simple_env.Simple(3)
S = np.zeros((mem_dim,s_dim))
R = np.zeros((mem_dim,1))
SPrime = np.zeros((mem_dim,s_dim))
cum_loss = 0
cum_sim_loss = 0
s = np.zeros((mb_dim,s_dim))
r = np.zeros((mb_dim,1))
rP = np.zeros((mb_dim,1))
rPP = np.zeros((mb_dim,1))
sPrime = np.zeros((mb_dim,s_dim))
import matplotlib.pyplot as plt
plt.ion()
'''grid of points'''
refresh = int(1e3)
for j in range(mem_dim):
    S[j] = env.observation_space.sample()
    SPrime[j],R[j],_ = env.get_transition(S[j],0)
print('MEM: ','pos: ',R[R>0].sum())
for j in range(mb_dim):
    s[j] = env.observation_space.sample()
    sP,r[j],_ = env.get_transition(s[j],0)
    sPP,rP[j],_ = env.get_transition(sP,0)
    _,rPP[j],_ = env.get_transition(sPP,0)
print('MB: ','pos: ',r[r>0].sum(),rP[rP>0].sum(),rPP[rPP>0].sum())
for i in range(int(1e4)):
    _,cur_loss,pred = sess.run([train_step,loss,pred_r],feed_dict={_s:s,_r:r,_rP:rP,_rPP:rPP,_mem_s:S,_mem_r:R,_mem_sPrime:SPrime})
    cum_loss += cur_loss
    if i % int(1e2) == 0:
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


