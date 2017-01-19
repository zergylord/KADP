import tensorflow as tf
sess = tf.Session()
from Utils.ops import *
import time
cur_time = time.clock()
import numpy as np
from Utils import simple_env
import gym
import os
import sys
if 'DISPLAY' in os.environ:
    display = True
    from matplotlib import pyplot as plt
    plt.ion()
else:
    display = False
    #sys.stdout = open("goat.txt", "w")
np.random.seed(111)
tf.set_random_seed(111)
print('hi',sess.run(tf.random_uniform((1,))),np.random.rand())
env = gym.make('Pendulum-v0')
env = simple_env.Simple(4)
#env = simple_env.Grid(one_hot=True)
#env = simple_env.Cycle(2,one_hot=True)
''' hyper parameters'''
s_dim = env.observation_space.shape
if not isinstance(s_dim,int):
    s_dim = np.prod(s_dim)
if env.action_space.__class__ == gym.spaces.box.Box:
    n_actions = 2
    A = [env.action_space.low,env.action_space.high]
else:
    n_actions = env.action_space.n
    A = list(range(n_actions))
print(n_actions)
hid_dim = 1000
z_dim = 1000
lr = 4e-3
mb_dim = 200
mem_dim = 400
n_viter = 10
n_viter_test = n_viter #can be higher to test for generalization
'''setup graph'''
eps = 1e-10
def make_encoder(inp,scope='encoder',reuse=False):
    #initial = tf.contrib.layers.xavier_initializer()
    initial = orthogonal_initializer()
    with tf.variable_scope(scope,reuse=reuse):
        last_hid = linear(inp,hid_dim,'hid1',tf.nn.relu,init=initial)
        #hid = linear(hid,hid_dim,'hid2',tf.nn.relu,init=initial)
        #last_hid = linear(hid,z_dim,'hid3',init=initial)
    return last_hid
#these losses assume scalar outputs (e.g. reward predictions)
def mse(o,t):
    return tf.reduce_mean(tf.square(o-t))
def bce(o,t):
    return -tf.reduce_mean(tf.log(tf.clip_by_value(o,eps,np.Inf))*t+tf.log(tf.clip_by_value(1-o,eps,np.Inf))*(1-t))
_s = tf.placeholder(tf.float32,shape=(None,s_dim))
_a = []
_r = []
_sPrime = []
for i in range(n_viter):
    _r.append(tf.placeholder(tf.float32,shape=(None,1)))
    _a.append(tf.placeholder(tf.float32,shape=(None,n_actions)))
    _sPrime.append(tf.placeholder(tf.float32,shape=(None,s_dim)))
lstm = tf.nn.rnn_cell.BasicLSTMCell(z_dim)
pred_r = []
r_loss = []
#open loop
for i in range(n_viter):
    if i > 0: tf.get_variable_scope().reuse_variables()
    if i == 0:
        #state = (tf.zeros((mb_dim,z_dim)),make_encoder(_s))
        state = (make_encoder(_s),tf.zeros((mb_dim,z_dim)))
    act_embed = linear(_a[i],z_dim,'act')
    output,state = lstm(act_embed,state)
    #pred_r.append(linear(linear(output,hid_dim,'hid1',tf.nn.relu),1,'r',tf.nn.sigmoid))
    pred_r.append(linear(output,1,'r',tf.nn.sigmoid))    
    tf.summary.histogram('r'+str(i),pred_r[i])
    r_loss.append(bce(pred_r[i],_r[i]))
    tf.summary.scalar('r loss '+str(i),r_loss[i])
'''
for i in range(n_viter):
    if i > 0: tf.get_variable_scope().reuse_variables()
    if i == 0:
        cur_s = _s
    else:
        cur_s = _sPrime[i-1]
    #act_embed = linear(_a[i],z_dim-2,'act')
    act_embed = _a[i]
    output = make_encoder(tf.concat_v2([act_embed,cur_s],axis=1))
    #pred_r.append(linear(linear(output,hid_dim,'hid1',tf.nn.relu),1,'r',tf.nn.sigmoid))
    pred_r.append(linear(output,1,'r',tf.nn.sigmoid))    
    tf.summary.histogram('r'+str(i),pred_r[i])
    r_loss.append(bce(pred_r[i],_r[i]))
    tf.summary.scalar('r loss '+str(i),r_loss[i])
'''
'''loss'''
loss = tf.add_n(r_loss)
tf.summary.scalar('net loss',loss)
optim = tf.train.AdamOptimizer(lr)
grads_and_vars = optim.compute_gradients(loss)
grad_summaries = [tf.summary.histogram('poo'+v.name,g) if g is not None else '' for g,v in grads_and_vars]
#capped_grads_and_vars = [(tf.clip_by_value(gv[0],-1,1),gv[1]) for gv in grads_and_vars]
train_step = optim.apply_gradients(grads_and_vars)

#check_op = tf.add_check_numerics_ops()
check_op = tf.no_op()
sess = tf.Session()
merged = tf.summary.merge_all()
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summary_dir', '/tmp/kadp', 'Summaries directory')
if tf.gfile.Exists(FLAGS.summary_dir):
    tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)
train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train',sess.graph)
sess.run(tf.global_variables_initializer())
S = []
R = []
SPrime = []
for i in range(n_actions):
    S.append(np.zeros((mem_dim,s_dim)))
    R.append(np.zeros((mem_dim,)))
    SPrime.append(np.zeros((mem_dim,s_dim)))
cum_loss = 0
cum_sim_loss = 0
s = np.zeros((mb_dim,s_dim))
'''grid of points'''
refresh = int(1e2)
bub_size = 50
num_steps = int(1e6)
#epsilon = np.concatenate([np.ones((int(2e3),)),np.linspace(1,.1,int(6e3)),np.ones((int(2e3),))*.1])
epsilon = np.ones((num_steps,))*.1
act = []
r = []
step_r = []
sPrime = []
cumr = 0
for i in range(n_viter):
    r.append(np.zeros((mb_dim,)))
    act.append(np.zeros((mb_dim,),dtype=np.int32))
    step_r.append(np.zeros((mb_dim,1)))
    sPrime.append(np.zeros((mb_dim,s_dim)))
oldest = np.zeros((n_actions))
for i in range(num_steps):
    feed_dict = {}
    for j in range(mb_dim):
        s[j] = env.reset()#env.observation_space.sample()
        cur_s = s[j].copy()
        for k in range(n_viter):
            act[k][j] = 0 #np.random.randint(n_actions)
            cur_s,r[k][j],_,_ = env.step(A[act[k][j]])#env.get_transition(cur_s,act[k][j])
            sPrime[k][j,:] = cur_s
    cumr += np.asarray(r).sum()
    #print('MB: ','pos: ',r[0][r[0]>0].sum(),r[1][r[1]>0].sum(),r[2][r[2]>0].sum())
    feed_dict[_s] = s
    for j in range(n_viter):
        feed_dict[_r[j]] = np.expand_dims(r[j],axis=-1)
        onehot = np.zeros((mb_dim,n_actions))
        onehot[np.arange(mb_dim),act] = 1.0
        feed_dict[_a[j]] = onehot
        feed_dict[_sPrime[j]] = sPrime[j]
    _,summary,_,cur_loss,*step_r = sess.run([check_op,merged,train_step,loss,*pred_r],feed_dict=feed_dict)
    #assert np.any(step_r[0] != step_r[1])
    train_writer.add_summary(summary)
    cum_loss += cur_loss
    if i % refresh == 0:
        cum_diff = 0
        for j in range(n_viter):
            cum_diff += (step_r[j] - r[j])
        cum_diff = np.squeeze(cum_diff)
        performance = cumr/refresh/mb_dim/n_viter
        print(i,'loss per sample: ',cum_loss/(n_viter)/refresh,'reward per step: ',performance,'net pred diff: ',cum_diff.sum(),'time: ',time.clock()-cur_time)
        cumr = 0
        cur_time = time.clock()
        cum_loss = 0
        #display stuff
        Xs = np.zeros((len(s),))
        Ys = np.zeros((len(s),))
        for state in range(len(s)):
            encoded = env.encode(s[state])
            Xs[state] = encoded[0]
            Ys[state] = encoded[1]
        plt.figure(1)
        plt.clf()
        plt.subplot(2, 3, 1)
        plt.scatter(Xs,Ys,s=bub_size,c=step_r[0])
        plt.subplot(2, 3, 2)
        plt.scatter(Xs,Ys,s=bub_size,c=step_r[int(n_viter/2)])
        plt.subplot(2, 3, 3)
        plt.scatter(Xs,Ys,s=bub_size,c=step_r[-1])
        plt.subplot(2, 3, 4)
        plt.scatter(Xs,Ys,s=bub_size,c=r[0])
        plt.subplot(2, 3, 5)
        plt.scatter(Xs,Ys,s=bub_size,c=r[int(n_viter/2)])
        plt.subplot(2, 3, 6)
        plt.scatter(Xs,Ys,s=bub_size,c=r[-1])
        plt.pause(.1)

