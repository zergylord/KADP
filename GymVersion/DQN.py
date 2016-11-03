import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from ops import *
import simple_env
import time
cur_time = time.clock()
in_dim = 2
hid_dim = 64
n_actions = 4
gamma = .9
epsilon = .1
lr = 1e-3
_s = tf.placeholder(tf.float32,shape=[None,in_dim])
_a = tf.placeholder(tf.int32,shape=[None,])
act_hot = tf.one_hot(_a,n_actions)
_r = tf.placeholder(tf.float32,shape=[None,])
_sPrime = tf.placeholder(tf.float32,shape=[None,in_dim])
_nt = tf.placeholder(tf.float32,shape=[None,])

def make_network(inp,scope='network',tied=False):
    act_func = tf.nn.softmax
    with tf.variable_scope(scope,reuse=tied):
        hid = linear(inp,hid_dim,'hid1',act_func)
        hid = linear(hid,hid_dim,'hid2',act_func)
        q = linear(hid,n_actions,'q')
    return q
cur_qs = make_network(_s)
next_qs = make_network(_sPrime,tied=True)
target = tf.stop_gradient(_r + _nt*gamma*tf.reduce_max(next_qs,1))
loss = tf.reduce_mean(tf.square(target-tf.reduce_sum(cur_qs*act_hot,1)))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

env = simple_env.Simple()
sess = tf.Session()
sess.run(tf.initialize_all_variables())
num_steps = int(1e5)
refresh = int(1e2)
mb_dim = 32
mb_s = np.zeros((mb_dim,in_dim))
mb_a = np.zeros((mb_dim,),dtype=np.int32)
mb_r = np.zeros((mb_dim,))
mb_sPrime = np.zeros((mb_dim,in_dim))
mb_nt = np.zeros((mb_dim,))
cumr = 0
cumloss = 0
plt.ion()
with tf.variable_scope('network/hid1',reuse=True):
    grad_op = tf.reduce_max(tf.gradients(loss,tf.get_variable('W')))
for i in range(num_steps):
    s = env.reset()
    for j in range(mb_dim):
        if np.random.rand() < epsilon:
            a = np.random.randint(n_actions)
        else:
            a = np.argmax(sess.run(cur_qs,feed_dict={_s:[s]})[0])
        sPrime,r,term,_ = env.step(a)
        cumr += r
        mb_s[j,:] = s
        mb_a[j] = a
        mb_r[j] = r
        mb_sPrime[j,:] = sPrime
        if term:
            mb_nt[j] = 0.0
            s = env.reset()
        else:
            mb_nt[j] = 1.0
            s = sPrime
    _,cur_loss,cur_grads = sess.run([train_step,loss,grad_op],
            feed_dict={_s:mb_s,_a:mb_a,_r:mb_r,_sPrime:mb_sPrime,_nt:mb_nt})
    cumloss += cur_loss
    if i % refresh == 0:
        print(cur_grads,'iter: ',i,'reward: ',1/(cumr/refresh/mb_dim+1e-10),'loss: ',cumloss,'time: ',time.clock()-cur_time)
        cur_time = time.clock()
        cumr = 0
        cumloss = 0

        x = np.linspace(-env.limit,env.limit,30)
        y = np.linspace(env.limit,-env.limit,30)
        xv, yv = np.meshgrid(x,y)
        count = 0
        test_s = np.zeros((900,in_dim))
        for xi in range(30):
            for yi in range(30):
                test_s[count,:] = np.asarray([xv[xi,yi],yv[xi,yi]])
                count +=1
        obs = simple_env.decode(test_s)
        q_grid = np.max(sess.run(cur_qs,feed_dict={_s:obs}),1)
        plt.scatter(test_s[:,0],test_s[:,1],c=np.log(q_grid),s=100)
        plt.pause(.001)

