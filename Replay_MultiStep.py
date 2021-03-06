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
if True or 'DISPLAY' in os.environ:
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
z_dim = 2 #5
lr = 4e-4
mb_dim = 200
mem_dim = 100
D_dim = int(1e5)
adapt_bandwidth = True
oracle = True
n_viter = 10
n_viter_test = n_viter #can be higher to test for generalization
'''setup replay buffer'''
D_S = np.zeros((n_actions,D_dim,s_dim))
D_SPrime = np.zeros((n_actions,D_dim,s_dim))
D_R = np.zeros((n_actions,D_dim))
'''setup graph'''
def make_encoder(inp,scope='encoder',reuse=False):
    #return inp
    #initial = tf.contrib.layers.xavier_initializer()
    initial = orthogonal_initializer()
    with tf.variable_scope(scope,reuse=reuse):
        hid = linear(inp,hid_dim,'hid1',tf.nn.relu,init=initial)
        hid = linear(hid,hid_dim,'hid2',tf.nn.relu,init=initial)
        last_hid = linear(hid,z_dim,'hid3',init=initial)
    #last_hid = tf.stop_gradient(last_hid)
    return last_hid
eps = 1e-10
if adapt_bandwidth:
    b = tf.exp(tf.Variable(-8.0,name='width'))
    tf.summary.scalar('kernel_width',b)
else:
    b = .01 # 8.3e-6
def kernel(z,mem_z,mother='rbf'):
    if mother == 'rbf':
        rbf = tf.exp(-tf.reduce_sum(tf.square(tf.expand_dims(z,-2)-mem_z),-1)/b) 
        normed = rbf/tf.clip_by_value(tf.reduce_sum(rbf,-1,keep_dims=True),eps,float("inf"))
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
    eps = 1e-10
    ratio = tf.clip_by_value(p,eps,np.Inf)/tf.clip_by_value(q,eps,np.Inf)
    log_ratio = tf.log(ratio)

    #log_ratio = tf.Print(log_ratio,[tf.reduce_sum(p,-1),tf.reduce_sum(q,-1),tf.reduce_max(log_ratio),tf.reduce_min(log_ratio),tf.reduce_min(ratio)])
    return tf.reduce_mean(tf.reduce_sum(p*log_ratio,-1))
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
U,_ = kernel(z,tf.gather(mem_z,_a[0]))
full_U,_ = kernel(z,mem_z_view)
tf.summary.histogram('U',full_U)
mem_sim = []
for a in range(n_actions):
    mem_sim.append(kernel(mem_zPrime_view,mem_z[a])[0])
full_mem_sim,_ = kernel(mem_zPrime_view,mem_z_view)
tf.summary.histogram('W',full_mem_sim)
pred_r = []
r_loss = []
s_loss = []
cur_sim = []
full_sim = []
for i in range(n_viter):
    cur_gamma = .9**i
    if i == 0:
        cur_sim.append(np.tile(np.eye(mem_dim,dtype=np.float32),[mb_dim,1,1]))
        full_sim.append(np.eye(n_actions*mem_dim,dtype=np.float32))
        weighted_R = tf.gather(_mem_r,_a[i])
    else:
        cur_mem_sim,_ = kernel(tf.gather(mem_zPrime,_a[i-1]),tf.expand_dims(tf.gather(mem_z,_a[i]),-3))
        cur_sim.append(tf.batch_matmul(cur_sim[i-1],cur_mem_sim))
        weighted_R = tf.reduce_sum(cur_sim[i]*tf.expand_dims(tf.gather(_mem_r,_a[i]),1),-1)
        full_sim.append(tf.matmul(full_sim[i-1],full_mem_sim))
        s_loss.append(kl(tf.matmul(full_U,full_sim[i]),simPrime[i-1]))
        #tf.summary.scalar('s loss '+str(i-1),s_loss[i-1])
    pred_r.append(tf.reduce_sum(U*weighted_R,-1))
    r_loss.append(bce(pred_r[i],_r[i])*cur_gamma)
    tf.summary.scalar('r loss '+str(i),r_loss[i])
'''variational stuff'''
one_step_loss = []
kl_loss = []
for i in range(n_viter):
    cur_U,_ = kernel(z,tf.gather(mem_z,_a[i]))
    cheat_pred_r = tf.reduce_sum(cur_U*tf.gather(_mem_r,_a[i]),-1)
    one_step_loss.append(bce(cheat_pred_r,_r[i]))
    kl_loss.append(kl(tf.stop_gradient(cur_U),tf.batch_matmul(tf.expand_dims(U,1),cur_sim[i])))
'''value'''
V = [tf.zeros((n_actions*mem_dim,))]
bell = _mem_r
for i in range(n_viter_test-1):
    new_V = tf.reduce_max(tf.reduce_sum(mem_sim*tf.expand_dims(bell,1),-1),0)
    V.append(new_V)
    bell = _mem_r+.9*tf.reshape(V[i],[n_actions,mem_dim])
mb_Q = tf.reduce_sum(U*tf.gather(bell,_a[0]),-1)
Q = []
for a in range(n_actions):
    U_a,_ = kernel(z,mem_z[a])
    Q.append(tf.reduce_sum(U_a*bell[a],-1))

'''loss'''
loss = tf.add_n(r_loss)#+tf.add_n(s_loss)*1e0
#loss = tf.add_n(one_step_loss) + tf.add_n(kl_loss)
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
#initial buffer
print('initial buffer filling...')
for j in range(D_dim):
    for a in range(n_actions):
        D_S[a,j,:] = env.reset()#env.observation_space.sample()
        D_SPrime[a,j,:],D_R[a,j],_,_ = env.step(A[a])#env.get_transition(S[a][j],a)
print('initial buffer filled')
oldest = np.zeros((n_actions))
for i in range(num_steps):
    #sample from buffer to fill out memories
    for a in range(n_actions):
        sample  = np.random.randint(D_dim,size=mem_dim)
        for j in range(mem_dim):
            S[a][j,:] = D_S[a,sample[j],:]
            SPrime[a][j,:] = D_SPrime[a,sample[j],:]
            R[a][j] = D_R[a,sample[j]]
    feed_dict = {}
    for a in range(n_actions):
        feed_dict[_mem_s[a]] = S[a]
        feed_dict[_mem_r[a]] = R[a]
        feed_dict[_mem_sPrime[a]] = SPrime[a]
    for j in range(mb_dim):
        if j == 0:
            cached_bell,cached_mem_z = sess.run([bell,mem_z],feed_dict=feed_dict)
            cached_dict = {bell:cached_bell}
            for a in range(n_actions):
                cached_dict[mem_z[a]] = cached_mem_z[a]
        s[j] = env.reset()#env.observation_space.sample()
        cur_s = s[j].copy()
        for k in range(n_viter):
            if oracle:
                act[k][j] = env.oracle_policy()
            elif np.random.rand() < epsilon[i]:
                act[k][j] = np.random.randint(n_actions)
            else:
                cached_dict[_s] = [cur_s]
                act[k][j] = np.argmax(np.asarray(sess.run(Q,feed_dict=cached_dict)),0)
            cur_s,r[k][j],_,_ = env.step(A[act[k][j]])#env.get_transition(cur_s,act[k][j])
            sPrime[k][j,:] = cur_s
    cumr += np.asarray(r).sum()
    #print('MB: ','pos: ',r[0][r[0]>0].sum(),r[1][r[1]>0].sum(),r[2][r[2]>0].sum())
    feed_dict[_s] = s
    for j in range(n_viter):
        feed_dict[_r[j]] = r[j]
        feed_dict[_a[j]] = act[j]
        feed_dict[_sPrime[j]] = sPrime[j]
    _,summary,_,cur_loss,mb_z,*step_r = sess.run([check_op,merged,train_step,loss,z,*pred_r],feed_dict=feed_dict)
    #assert np.any(step_r[0] != step_r[1])
    train_writer.add_summary(summary)
    cum_loss += cur_loss
    #add new memories
    for j in range(mb_dim):
        a = act[0][j]
        ind = int(oldest[a])
        D_S[a,ind,:] = s[j].copy()
        for k in range(n_viter):
            D_R[a,ind] = r[k][j]
            D_SPrime[a,ind,:] = sPrime[k][j].copy()
            if k < (n_viter-1):
                D_S[a,ind,:] = sPrime[k+1][j].copy()
                oldest[a] = (oldest[a]+ 1) % D_dim
                a = act[k+1][j]
                ind = int(oldest[a])
    if i % refresh == 0:
        for j in range(n_actions):
            print('MEM: ','pos: ',R[j][R[j]>0].sum())
        value = sess.run(mb_Q,feed_dict=feed_dict)
        #print(R.sum())
        cum_diff = 0
        for j in range(n_viter):
            cum_diff += (step_r[j] - r[j])
        cum_diff = np.squeeze(cum_diff)
        performance = cumr/refresh/mb_dim/n_viter
        print(i,'loss per sample: ',cum_loss/(n_viter),'reward per step: ',performance,'net pred diff: ',cum_diff.sum(),'time: ',time.clock()-cur_time)
        #TODO: should reset memory buffer
        if performance > .1: #task specific to 10x10 grid
            '''
            print('+++++++++++winner+++++++++++')
            env.gen_goal()
            print('initial buffer filling...')
            for j in range(D_dim):
                for a in range(n_actions):
                    D_S[a,j,:] = env.reset()#env.observation_space.sample()
                    D_SPrime[a,j,:],D_R[a,j],_,_ = env.step(A[a])#env.get_transition(S[a][j],a)
            print('initial buffer filled')
            '''
        cumr = 0
        cur_time = time.clock()
        cum_loss = 0
        if display:
            if s_dim == 2 or env.__class__ == simple_env.Grid: 
                '''
                Xs = env.encode(s[:,0])
                Ys = env.encode(s[:,1])
                '''
                Xs = np.zeros((len(s),))
                Ys = np.zeros((len(s),))
                for state in range(len(s)):
                    encoded = env.encode(s[state])
                    Xs[state] = encoded[0]
                    Ys[state] = encoded[1]
                plt.figure(1)
                plt.clf()
                plt_rows,plt_cols = 3,4
                #predicted rewards
                plt.subplot(plt_rows,plt_cols, 1)
                plt.scatter(Xs,Ys,s=bub_size,c=step_r[0])
                plt.subplot(plt_rows,plt_cols, 2)
                plt.scatter(Xs,Ys,s=bub_size,c=step_r[int(n_viter/2)])
                plt.subplot(plt_rows,plt_cols, 3)
                plt.scatter(Xs,Ys,s=bub_size,c=step_r[-1])
                plt.subplot(plt_rows,plt_cols, 4)
                #real rewards
                plt.scatter(Xs,Ys,s=bub_size,c=value)#np.log(value[:,0]+1e-10))
                plt.subplot(plt_rows,plt_cols, 5)
                plt.scatter(Xs,Ys,s=bub_size,c=r[0])
                plt.subplot(plt_rows,plt_cols, 6)
                plt.scatter(Xs,Ys,s=bub_size,c=r[int(n_viter/2)])
                plt.subplot(plt_rows,plt_cols, 7)
                plt.scatter(Xs,Ys,s=bub_size,c=r[-1])
                #latent viz
                zax = plt.subplot(plt_rows,plt_cols,9)
                plt.scatter(cached_mem_z[0][:,0],cached_mem_z[0][:,1],c=cached_bell[0])
                plt.subplot(plt_rows,plt_cols,10,sharex=zax,sharey=zax)
                plt.scatter(mb_z[:,0],mb_z[:,1],c=value)
                plt.subplot(plt_rows,plt_cols,11)#,sharex=zax,sharey=zax)
                #combined = np.concatenate([cached_mem_z[0],mb_z])
                combined = np.concatenate([S[0],s])
                combined_colors = np.zeros([combined.shape[0],])
                combined_colors[mb_dim:] = 1
                plt.scatter(combined[:,0],combined[:,1],c=combined_colors)

                plt.figure(3)
                plt.clf()
                offX = .5*env.radius*np.cos(env.rad_inc*np.arange(n_actions))
                offY = .5*env.radius*np.sin(env.rad_inc*np.arange(n_actions))
                plt.hold(True)
                mb_q_values = np.asarray(sess.run(Q,feed_dict=feed_dict))
                mb_values = np.max(mb_q_values,0)
                for action in range(n_actions):
                    mask = np.argmax(mb_q_values,0) == action
                    plt.scatter(Xs[mask]+offX[action],Ys[mask]+offY[action],s=bub_size/2)
                plt.scatter(Xs,Ys,s=bub_size,c=(mb_values))
                axes = plt.gca()
                '''
                axes.set_xlim([-env.limit,env.limit])
                axes.set_ylim([-env.limit,env.limit])
                '''
                plt.hold(False)
            elif env.__class__ == simple_env.Cycle:
                qvals = [np.zeros(s_dim)]*n_actions
                feed_dict[_s] = np.eye(s_dim)
                qvals = sess.run(Q,feed_dict=feed_dict)

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
'''
plt.ioff()
plt.show()
'''

