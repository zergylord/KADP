import tensorflow as tf
sess = tf.Session()
from Utils.ops import *
import time
cur_time = time.clock()
import numpy as np
from Utils import simple_env
'''
np.random.seed(111)
tf.set_random_seed(111)
'''
print('hi',sess.run(tf.random_uniform((1,))),np.random.rand())
env = simple_env.Cycle(2,one_hot=True)
''' hyper parameters'''
s_dim = env.observation_space.shape
n_actions = env.action_space.n
print(n_actions)
hid_dim = 128
z_dim = 64
lr = 1e-4
mb_dim = 200
mem_dim = 100
n_viter = 5
n_viter_test = 20
'''setup graph'''
def make_encoder(inp,scope='encoder',reuse=False):
    #initial = tf.contrib.layers.xavier_initializer()
    initial = orthogonal_initializer()
    with tf.variable_scope(scope,reuse=reuse):
        hid = linear(inp,hid_dim,'hid1',tf.nn.relu,init=initial)
        hid = linear(hid,hid_dim,'hid2',tf.nn.relu,init=initial)
        last_hid = linear(hid,z_dim,'hid3',init=initial)
    return last_hid
eps = 1e-10
def kernel(z,mem_z,mother='rbf'):
    if mother == 'rbf':
        b = 1
        rbf = tf.exp(-tf.reduce_sum(tf.square(tf.expand_dims(z,-2)-mem_z),-1)/b) 
        normed = rbf/tf.reduce_sum(rbf,-1,keep_dims=True)
        return normed,rbf
    elif mother == 'dot':
        dot = tf.reduce_sum(tf.expand_dims(z,-2)*mem_z,-1)
        return tf.nn.softmax(dot),dot
        #return (dot+1)/tf.reduce_sum(dot+1),dot
    elif mother == 'cosine':
        inv_mag_z = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(z),-1,keep_dims=True),eps,float("inf")))
        inv_mag_mem_z = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(mem_z),-1,keep_dims=True),eps,float("inf")))
        dot = tf.reduce_sum(tf.expand_dims(z,-2)*mem_z,-1)*inv_mag_z*inv_mag_mem_z
        #return tf.nn.softmax(dot),dot
        return (dot+1)/tf.reduce_sum(dot+1),dot
    else:
        print('nope')
def kl(p,q):
    return tf.reduce_mean(tf.reduce_sum(p*tf.log(p/q),-1))
def mse(o,t):
    return tf.reduce_mean(tf.reduce_sum(tf.square(o-t),-1))
_s = tf.placeholder(tf.float32,shape=(None,s_dim))
_a = []
_r = []
_sPrime = []
for i in range(n_viter):
    _r.append(tf.placeholder(tf.float32,shape=(None,)))
    _a.append(tf.placeholder(tf.int32,shape=(None,)))
    _sPrime.append(tf.placeholder(tf.float32,shape=(None,s_dim)))
_mem_s = []
_mem_r = []
_mem_sPrime = []
mem_z = []
mem_zPrime = []
for a in range(n_actions):
    _mem_s.append(tf.placeholder(tf.float32,shape=(None,s_dim)))
    _mem_r.append(tf.placeholder(tf.float32,shape=(None,)))
    _mem_sPrime.append(tf.placeholder(tf.float32,shape=(None,s_dim)))
    '''embedings'''
    if a == 0:
        mem_z.append(make_encoder(_mem_s[a]))
    else:
        mem_z.append(make_encoder(_mem_s[a],reuse=True))
    mem_zPrime.append(make_encoder(_mem_sPrime[a],reuse=True))
mem_z_view = tf.concat(0,mem_z)
mem_zPrime_view = tf.concat(0,mem_zPrime)
z = make_encoder(_s,reuse=True)
zPrime = []
simPrime = []
for i in range(n_viter):
    zPrime.append(make_encoder(_sPrime[i],reuse=True))
    simPrime.append(kernel(zPrime[i],mem_z_view)[0])

'''similarity'''
sim,_ = kernel(z,tf.gather(mem_z,_a[0]))
mem_sim = []
for a in range(n_actions):
    mem_sim.append(kernel(mem_zPrime_view,mem_z[a])[0])
#TODO: get just the correct action from mem_zPrime for mem_sim in the multistep r
pred_r = []
r_loss = []
s_loss = []
cur_sim = []
for i in range(n_viter):
    cur_gamma = 1.0**i
    if i == 0:
        cur_sim.append(np.tile(np.eye(mem_dim,dtype=np.float32),[mb_dim,1,1]))
        weighted_R = tf.gather(_mem_r,_a[i])
    else:
        cur_mem_sim,_ = kernel(tf.gather(mem_zPrime,_a[i-1]),tf.expand_dims(tf.gather(mem_z,_a[i]),-3))
        cur_sim.append(tf.batch_matmul(cur_sim[i-1],cur_mem_sim))
        weighted_R = tf.reduce_sum(cur_sim[i]*tf.expand_dims(tf.gather(_mem_r,_a[i]),1),-1)
    pred_r.append(tf.reduce_sum(cur_gamma*sim*weighted_R,-1))
    #s_loss.append(kl(tf.matmul(sim,cur_sim[i+1]),simPrime[i]))
    #tf.scalar_summary('s loss '+str(i),s_loss[i])
    r_loss.append(mse(pred_r[i],_r[i]))
    tf.scalar_summary('r loss '+str(i),r_loss[i])
'''value'''
V = [tf.zeros((n_actions*mem_dim,))]
bell = _mem_r
for i in range(n_viter_test-1):
    new_V = tf.reduce_max(tf.reduce_sum(mem_sim*tf.expand_dims(bell,1),-1),0)
    V.append(new_V)
    bell = _mem_r+.9*tf.reshape(V[i],[n_actions,mem_dim])
mb_V = tf.reduce_sum(sim*tf.gather(bell,_a[0]),-1)
Q = []
grid_s = np.eye(s_dim)
grid_z = make_encoder(tf.constant(grid_s,dtype=tf.float32),reuse=True)
for a in range(n_actions):
    grid_sim,_ = kernel(grid_z,mem_z[a])
    Q.append(tf.reduce_sum(grid_sim*bell[a],-1))

'''loss'''
loss = tf.add_n(r_loss)#+tf.add_n(s_loss)*1e-3
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
import matplotlib.pyplot as plt
plt.ion()
'''grid of points'''
refresh = int(1e2)
bub_size = 50
act = []
r = []
step_r = []
sPrime = []
for i in range(n_viter):
    r.append(np.zeros((mb_dim,)))
    act.append(np.zeros((mb_dim,)))
    step_r.append(np.zeros((mb_dim,1)))
    sPrime.append(np.zeros((mb_dim,s_dim)))
for i in range(int(1e7)):
    for j in range(mb_dim):
        s[j] = env.observation_space.sample()
        cur_s = s[j]
        for k in range(n_viter):
            act[k][j] = np.random.randint(n_actions)
            cur_s,r[k][j],_ = env.get_transition(cur_s,act[k][j])
            sPrime[k][j,:] = cur_s
    #print('MB: ','pos: ',r[0][r[0]>0].sum(),r[1][r[1]>0].sum(),r[2][r[2]>0].sum())
    for j in range(mem_dim):
        for a in range(n_actions):
            S[a][j] = env.observation_space.sample()
            SPrime[a][j],R[a][j],_ = env.get_transition(S[a][j],a)
    #print('MEM: ','pos: ',R[R>0].sum())
    feed_dict = {_s:s}
    for a in range(n_actions):
        feed_dict[_mem_s[a]] = S[a]
        feed_dict[_mem_r[a]] = R[a]
        feed_dict[_mem_sPrime[a]] = SPrime[a]
    for j in range(n_viter):
        feed_dict[_r[j]] = r[j]
        feed_dict[_a[j]] = act[j]
        feed_dict[_sPrime[j]] = sPrime[j]
    summary,_,cur_loss,*step_r = sess.run([merged,train_step,loss,*pred_r],feed_dict=feed_dict)
    #assert np.any(step_r[0] != step_r[1])
    train_writer.add_summary(summary)
    cum_loss += cur_loss
    if i % refresh == 0:
        qvals = [np.zeros(s_dim)]*n_actions
        value,*qvals = sess.run([mb_V,*Q],feed_dict=feed_dict)
        #print(R.sum())
        cum_diff = 0
        for j in range(n_viter):
            cum_diff += (step_r[j] - r[j])
        cum_diff = np.squeeze(cum_diff)
        print(i,cum_loss,cum_diff.sum(),time.clock()-cur_time)
        cur_time = time.clock()
        cum_loss = 0
        if s_dim == 2: 
            Xs = env.encode(s[:,0])
            Ys = env.encode(s[:,1])
            plt.figure(1)
            plt.clf()
            plt.subplot(2, 2, 1)
            plt.scatter(Xs,Ys,s=bub_size,c=step_r[0])
            plt.subplot(2, 2, 2)
            plt.scatter(Xs,Ys,s=bub_size,c=step_r[int(n_viter/2)])
            plt.subplot(2, 2, 3)
            plt.scatter(Xs,Ys,s=bub_size,c=step_r[-1])
            plt.subplot(2, 2, 4)
            plt.scatter(Xs,Ys,s=bub_size,c=value[:,0])#np.log(value[:,0]+1e-10))
            plt.figure(2)
            plt.clf()
            plt.subplot(2, 2, 1)
            plt.scatter(Xs,Ys,s=bub_size,c=r[0])
            plt.subplot(2, 2, 2)
            plt.scatter(Xs,Ys,s=bub_size,c=r[int(n_viter/2)])
            plt.subplot(2, 2, 3)
            plt.scatter(Xs,Ys,s=bub_size,c=r[-1])
            plt.figure(3)
            plt.clf()
            plt.scatter(Xs,Ys,s=bub_size,c=(cum_diff))
        else:
            pos = env.encode(s)
            plt.figure(1)
            plt.clf()
            plt.subplot(2, 2, 1)
            plt.bar(pos,step_r[0])
            plt.subplot(2, 2, 2)
            plt.bar(pos,step_r[int(n_viter/2)])
            plt.subplot(2, 2, 3)
            plt.bar(pos,step_r[-1])
            plt.subplot(2, 2, 4)
            plt.bar(pos,value)

            plt.figure(2)
            plt.clf()
            plt.subplot(3,1,1)
            plt.bar(np.arange(s_dim),qvals[0])
            plt.subplot(3,1,2)
            plt.bar(np.arange(s_dim),qvals[1])
            plt.subplot(3,1,3)
            plt.bar(np.arange(s_dim),qvals[1]-qvals[0])

        plt.pause(.01)
    #env.gen_goal()
plt.ioff()
plt.show()

