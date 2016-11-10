import tensorflow as tf
sess = tf.Session()
from ops import *
import time
import numpy as np
import simple_env
'''
np.random.seed(111)
tf.set_random_seed(111)
'''
print('hi',sess.run(tf.random_uniform((1,))),np.random.rand())
''' hyper parameters'''
s_dim = 2
hid_dim = 100
z_dim = 2
lr = 1e-4
mb_dim = 100
mem_dim = 1000
'''setup graph'''
def make_network(inp,scope='network',reuse=False):
    #initial = tf.contrib.layers.xavier_initializer()
    initial = orthogonal_initializer()
    with tf.variable_scope(scope,reuse=reuse):
        hid = linear(inp,hid_dim,'hid1',tf.nn.relu,init=initial)
        #hid = linear(hid,hid_dim,'hid2',tf.nn.relu,init=initial)
        last_hid = linear(hid,z_dim,'hid3',init=initial)
    return last_hid
def kernel(z,mem_z,mother='rbf'):
    if mother == 'rbf':
        b = 1
        rbf = tf.exp(-tf.reduce_sum(tf.square(tf.expand_dims(z,1)-mem_z),-1)/b) 
        normed = rbf/tf.reduce_sum(rbf,-1,keep_dims=True)
        rbf = tf.Print(rbf,[tf.reduce_sum(normed,-1)])
        return normed,rbf
    elif mother == 'dot':
        dot = tf.reduce_sum(tf.expand_dims(z,1)*mem_z,-1)
        return tf.nn.softmax(dot),dot
    else:
        print('nope')
def kl(p,q):
    return tf.reduce_mean(tf.reduce_sum(p*tf.log(p/q),-1))
_s = tf.placeholder(tf.float32,shape=(None,s_dim))
_sPrime = tf.placeholder(tf.float32,shape=(None,s_dim))
_mem_s = tf.placeholder(tf.float32,shape=(None,s_dim))
_mem_sPrime = tf.placeholder(tf.float32,shape=(None,s_dim))

mem_z = make_network(_mem_s)
mem_zPrime = make_network(_mem_sPrime,reuse=True)
z = make_network(_s,reuse=True)
sim,_ = kernel(z,mem_z)
mem_sim,unnorm = kernel(mem_zPrime,mem_z)
pred_sPrime_sim = tf.matmul(sim,mem_sim)
zPrime = make_network(_sPrime,reuse=True)
sPrime_sim,_ = kernel(zPrime,mem_z)
#sPrime_sim = tf.stop_gradient(sPrime_sim)
#loss = tf.reduce_mean(tf.reduce_sum(tf.square(pred_sPrime_sim-sPrime_sim),-1))
kl_loss = kl(pred_sPrime_sim,sPrime_sim)
#kl_loss = kl(sPrime_sim,pred_sPrime_sim)
cross_sim = tf.reduce_mean(tf.reduce_sum(tf.square(unnorm),1))
sim_loss = tf.square(cross_sim - 1000)
loss = kl_loss # + sim_loss/(1e5) 
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
#train_step = tf.no_op()

sess = tf.Session()
sess.run(tf.initialize_all_variables())
env = simple_env.Simple(3)
S = np.zeros((mem_dim,s_dim))
SPrime = np.zeros((mem_dim,s_dim))
cum_loss = 0
cum_kl_loss = 0
cum_sim_loss = 0
s = np.zeros((mb_dim,s_dim))
sPrime = np.zeros((mb_dim,s_dim))
import matplotlib.pyplot as plt
plt.ion()
'''grid of points'''
side = int(np.sqrt(mb_dim))
x = np.linspace(-env.limit,env.limit,side)
y = np.linspace(env.limit,-env.limit,side)
xv, yv = np.meshgrid(x,y)
count = 0
for xi in range(side):
    for yi in range(side):
        s[count,:] = np.asarray([xv[xi,yi],yv[xi,yi]])
        s[count,:] = simple_env.decode(s[count])
        a = env.action_space.sample() #ignoring actions for now -- passive dynamics
        sPrime[count,:],_,_ = env.get_transition(s[count],a)
        count +=1
for j in range(mem_dim):
    S[j] = env.observation_space.sample()
    a = env.action_space.sample() #ignoring actions for now -- passive dynamics
    SPrime[j],_,_ = env.get_transition(S[j],a)
feed_dict={_s:s,_sPrime:sPrime,_mem_s:S,_mem_sPrime:SPrime}
prox = sess.run(sim,feed_dict={_s:s,_sPrime:sPrime,_mem_s:s,_mem_sPrime:sPrime})
refresh = int(1e3)
if True:
    for i in range(int(1e5)):
        '''
        for j in range(mb_dim):
            s[j] = env.observation_space.sample()
            a = env.action_space.sample() #ignoring actions for now -- passive dynamics
            sPrime[j],_,_ = env.get_transition(s[j],a)
        '''
        _,cur_loss,cur_kl_loss,cur_sim_loss,latent,bad = sess.run([train_step,loss,kl_loss,sim_loss,z,cross_sim],feed_dict=feed_dict)
        cum_loss += cur_loss
        cum_kl_loss += cur_kl_loss
        cum_sim_loss += cur_sim_loss
        if i % refresh == 0:
            print(i,cum_loss,cum_kl_loss,cum_sim_loss,bad)
            cum_loss = 0
            cum_kl_loss = 0
            cum_sim_loss = 0
            if z_dim == 2:
                plt.figure(0)
                Xs = latent[:,0]
                Ys = latent[:,1]
                plt.clf()
                plt.scatter(Xs,Ys,s=100)
                axes = plt.gca()
                '''
                axes.set_xlim([-1,1])
                axes.set_ylim([-1,1])
                '''
            plt.figure(1)
            prox = sess.run(sim,feed_dict={_s:s,_sPrime:sPrime,_mem_s:s,_mem_sPrime:sPrime})
            plt.clf()
            plt.imshow(prox)
            plt.pause(.01)
    plt.figure(2)
    Xs = simple_env.encode(s[:,0])
    Ys = simple_env.encode(s[:,1])
    for j in range(mb_dim):
        plt.clf()
        plt.scatter(Xs,Ys,s=100,c=prox[j])
        axes = plt.gca()
        axes.set_xlim([-env.limit,env.limit])
        axes.set_ylim([-env.limit,env.limit])
        plt.pause(.001)

