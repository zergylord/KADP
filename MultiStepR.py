import tensorflow as tf
sess = tf.Session()
from Utils.ops import *
import time
cur_time = time.clock()
import numpy as np
from Utils import simple_env
np.random.seed(111)
tf.set_random_seed(111)
print('hi',sess.run(tf.random_uniform((1,))),np.random.rand())
''' hyper parameters'''
s_dim = 2
hid_dim = 64
z_dim = 10
lr = 1e-3
mb_dim = 300
mem_dim = 300
n_viter = 10
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
_r = []
for i in range(n_viter):
    _r.append(tf.placeholder(tf.float32,shape=(None,1)))
_mem_s = tf.placeholder(tf.float32,shape=(None,s_dim))
_mem_r = tf.placeholder(tf.float32,shape=(None,1))
_mem_sPrime = tf.placeholder(tf.float32,shape=(None,s_dim))
'''embedings'''
mem_z = make_encoder(_mem_s)
mem_zPrime = make_encoder(_mem_sPrime,reuse=True)
z = make_encoder(_s,reuse=True)
'''similarity'''
sim,_ = kernel(z,mem_z)
mem_sim,_ = kernel(mem_zPrime,mem_z)
pred_r = []
r_loss = []
cur_sim = []
cur_sim.append(np.eye(mem_dim,dtype=np.float32))
for i in range(n_viter):
    cur_gamma = 1.0**i
    pred_r.append(cur_gamma*tf.matmul(sim,tf.matmul(cur_sim[i],_mem_r)))
    cur_sim.append(tf.matmul(cur_sim[i],mem_sim))
    r_loss.append(mse(pred_r[i],_r[i]))
    tf.scalar_summary('r loss '+str(i),r_loss[i])
'''value'''
V = [tf.zeros((mem_dim,1))]
for i in range(n_viter-1):
    bell = _mem_r+.9*V[i]
    new_V = tf.matmul(mem_sim,bell)
    V.append(new_V)
bell = _mem_r+.9*V[-1]
mb_V = tf.matmul(sim,bell)

'''loss'''
loss = tf.add_n(r_loss)
tf.scalar_summary('net loss',loss)
optim = tf.train.AdamOptimizer(lr)
grads_and_vars = optim.compute_gradients(loss)
grad_summaries = [tf.histogram_summary('poo'+v.name,g) if g is not None else '' for g,v in grads_and_vars]
train_step = optim.apply_gradients(grads_and_vars)

#check_op = tf.add_check_numerics_ops()
check_op = tf.no_op()
sess = tf.Session()
merged = tf.merge_all_summaries()
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summary_dir', '/tmp/kadp', 'Summaries directory')
if tf.gfile.Exists(FLAGS.summary_dir):
    tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)
train_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/train',sess.graph)
sess.run(tf.initialize_all_variables())
env = simple_env.Simple(3)
S = np.zeros((mem_dim,s_dim))
R = np.zeros((mem_dim,1))
SPrime = np.zeros((mem_dim,s_dim))
cum_loss = 0
cum_sim_loss = 0
s = np.zeros((mb_dim,s_dim))
rP = np.zeros((mb_dim,1))
rPP = np.zeros((mb_dim,1))
sPrime = np.zeros((mb_dim,s_dim))
import matplotlib.pyplot as plt
plt.ion()
'''grid of points'''
refresh = int(1e3)
bub_size = 50
r = []
step_r = []
for i in range(n_viter):
    r.append(np.zeros((mb_dim,1)))
    step_r.append(np.zeros((mb_dim,1)))
for j in range(mem_dim):
    S[j] = env.observation_space.sample()
    SPrime[j],R[j],_ = env.get_transition(S[j],0)
print('MEM: ','pos: ',R[R>0].sum())
for i in range(int(1e5)):
    for j in range(mb_dim):
        s[j] = env.observation_space.sample()
        cur_s = s[j]
        for k in range(n_viter):
            cur_s,r[k][j],_ = env.get_transition(cur_s,0)
    #print('MB: ','pos: ',r[0][r[0]>0].sum(),r[1][r[1]>0].sum(),r[2][r[2]>0].sum())
    feed_dict = {_s:s,_mem_s:S,_mem_r:R,_mem_sPrime:SPrime}
    for j in range(n_viter):
        feed_dict[_r[j]] = r[j]
    summary,_,cur_loss,*step_r = sess.run([merged,train_step,loss,*pred_r],feed_dict=feed_dict)
    #assert np.any(step_r[0] != step_r[1])
    train_writer.add_summary(summary)
    cum_loss += cur_loss
    if i % refresh == 0:
        value = sess.run(mb_V,feed_dict=feed_dict)
        #print(R.sum())
        cum_diff = 0
        for j in range(n_viter):
            cum_diff += (step_r[j] - r[j])
        cum_diff = np.squeeze(cum_diff)
        print(i,cum_loss,cum_diff.sum(),time.clock()-cur_time)
        cur_time = time.clock()
        cum_loss = 0
        Xs = env.encode(s[:,0])
        Ys = env.encode(s[:,1])
        plt.figure(1)
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.scatter(Xs,Ys,s=bub_size,c=step_r[0])
        plt.subplot(2, 2, 2)
        plt.scatter(Xs,Ys,s=bub_size,c=step_r[1])
        plt.subplot(2, 2, 3)
        plt.scatter(Xs,Ys,s=bub_size,c=step_r[2])
        plt.subplot(2, 2, 4)
        plt.scatter(Xs,Ys,s=bub_size,c=value[:,0])#c=np.log(value[:,0]+1e-10))
        plt.figure(2)
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.scatter(Xs,Ys,s=bub_size,c=r[0])
        plt.subplot(2, 2, 2)
        plt.scatter(Xs,Ys,s=bub_size,c=r[1])
        plt.subplot(2, 2, 3)
        plt.scatter(Xs,Ys,s=bub_size,c=r[2])
        plt.figure(3)
        plt.clf()
        plt.scatter(Xs,Ys,s=bub_size,c=(cum_diff))
        plt.pause(.01)
plt.ioff()
plt.show()


