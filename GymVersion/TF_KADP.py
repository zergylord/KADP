import time
import numpy as np
import tensorflow as tf
np.random.seed(1)
tf.set_random_seed(1)
print(np.random.rand())
sess = tf.Session()
foo = sess.run(tf.random_uniform((1,)))
print('hi',foo)
import simple_env
from ops import *
from matplotlib import pyplot as plt
plt.ion()
from scipy.spatial.distance import cdist,pdist
class KADP(object):
    def make_network(self,inp,scope='network',tied=False):
        with tf.variable_scope(scope,reuse=tied):
            hid = linear(inp,self.hid_dim,'hid1',tf.nn.relu)
            #hid = linear(hid,self.hid_dim,'hid2',tf.nn.relu)
            last_hid = linear(hid,self.s_dim,'hid3')
        return last_hid
    def kernel(self,x,X):
        if not self.net_exists:
            print('new network!')
            x = self.make_network(x)
            self.net_exists = True
        else:
            x = self.make_network(x,tied=True)
        if len(X.shape) > 2:
            true_shape = X.shape
            X = self.make_network(tf.reshape(X,[-1,self.s_dim]),tied=True)
            X = tf.reshape(X,true_shape)
        else:
            X = self.make_network(X,tied=True)
        #2x2 4x100x2
        '''Gaussian'''
        '''
        dist = tf.sqrt(tf.reduce_sum(tf.square(x-X),-1))
        sim = tf.exp(-tf.clip_by_value(dist,0,10))
        '''
        '''dot-product'''
        '''reshape to (n_actions,mb_dim,samples_per_action,s_dim)'''
        x = tf.expand_dims(tf.expand_dims(x,1),0)
        X = tf.expand_dims(X,1)
        inv_mag = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(X),-1,keep_dims=True),eps,float("inf")))
        sim = tf.squeeze(tf.reduce_sum(X*x,-1,keep_dims=True)*inv_mag)
        k_sim,k_inds = tf.nn.top_k(sim,k=self.k,sorted=False)

        return k_sim,k_inds
    def get_value(self,s):
        weights,inds = self.kernel(s,self.S)
        normed_weights = tf.nn.softmax(weights)
        row_inds = self.row_offsets+inds
        R_ = tf.gather(self.R_view,row_inds)
        NT_ = tf.gather(self.NT_view,row_inds)
        V_ = tf.gather(self.cur_V,row_inds)
        q_vals = tf.reduce_sum(normed_weights*(R_+NT_*self.gamma*V_),-1)
        if self.softmax:
            val = tf.reduce_sum(tf.nn.softmax(q_vals,dim=0)*q_vals,0)
        else:
            val = tf.reduce_max(q_vals,0)
        action = tf.argmax(q_vals,0) #this is wasteful!
        return val,action
    def __init__(self,env,W_and_NNI = None):
        self.net_exists = False
        self.n_actions = env.action_space.n
        self.samples_per_action = 100
        self.n_samples = self.n_actions*self.samples_per_action
        #for converting inds for a particular action to row inds
        self.row_offsets = np.expand_dims(np.expand_dims(np.arange(self.n_actions)*self.samples_per_action,-1),-1) 
        
        self.s_dim = 2
        self.k = 90
        self.b = 1e1
        self.hid_dim = 128
        self.lr = 1e-4
        self.softmax = True
        self.change_actions = True
        self.gamma = .9
        '''create dataset'''
        self.S = np.zeros((self.n_actions,self.samples_per_action,self.s_dim)).astype(np.float32())
        self.SPrime = np.zeros((self.n_actions,self.samples_per_action,self.s_dim)).astype(np.float32())
        self.SPrime_view = self.SPrime.reshape(-1,self.s_dim)
        self.R = np.zeros((self.n_actions,self.samples_per_action)).astype(np.float32())
        self.R_view = self.R.reshape(-1)
        self.NT = np.zeros((self.n_actions,self.samples_per_action)).astype(np.float32())
        self.NT_view = self.NT.reshape(-1)
        ''' these should be tensors
        self.V = np.zeros((self.n_actions,self.samples_per_action))
        self.V_view = self.V.reshape(-1)
        '''
        for a in range(self.n_actions):
            for i in range(self.samples_per_action):
                s = env.observation_space.sample()
                sPrime,r,term = env.get_transition(s,a)
                self.S[a,i] = s
                self.SPrime[a,i] = sPrime
                self.R[a,i] = r
                self.NT[a,i] = np.float32(not term)
        ''' create similarity sparse matrix'''
        if W_and_NNI == None:
            self.W,inds = self.kernel(self.SPrime_view,self.S)
            self.NNI = self.row_offsets+inds
        else:
            self.W,self.NNI = W_and_NNI
        '''create computation graph'''
        normed_W = tf.nn.softmax(self.W)
        V = [tf.zeros((self.n_samples,),dtype=tf.float32)]
        self.cur_V = V[-1]
        inds = self.NNI
        R_ = tf.gather(self.R_view,inds)
        NT_ = tf.gather(self.NT_view,inds)
        for t in range(1):
            V_ = tf.gather(V[t],inds)
            q_vals = tf.reduce_sum(normed_W*(R_+NT_*self.gamma*V_),-1)
            if self.softmax:
                V.append(tf.reduce_sum(tf.nn.softmax(q_vals,dim=0)*q_vals,0))
            else:
                V.append(tf.reduce_max(q_vals,0))
            self.cur_V = V[-1]
            '''
            foo,_ = self.get_value(self.SPrime_view[0])
            tf.Assert(foo == V[-1][0],[foo,V[-1][0]])
            '''
        '''NOTE: This TD error is incorrect for exploratory actions'''
        self._s = tf.placeholder(tf.float32,shape=(None,self.s_dim,))
        self._r = tf.placeholder(tf.float32,shape=(None,1,))
        self._sPrime = tf.placeholder(tf.float32,shape=(None,self.s_dim,))
        self._nt = tf.placeholder(tf.float32,shape=(None,1,))
        self.init_value,_ = self.get_value(self._s)
        self.final_value,_ = self.get_value(self._sPrime)
        target = tf.stop_gradient(self._r + self._nt*self.gamma*self.final_value)
        self.loss = tf.reduce_mean(tf.square(target-self.init_value))
        with tf.variable_scope('network/hid1',reuse=True):
            net_weights = tf.get_variable('W')
        self.get_grads = tf.gradients(self.loss,net_weights)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
env = simple_env.Simple()
agent = KADP(env)
sess.run(tf.initialize_all_variables())
cur_time = time.clock()
epsilon = .1

cumloss = 0
cumgrads = 0
num_steps = int(1e6)
refresh = int(1e3)
mb_cond = 1
mb_dim = 900
mb_s = np.zeros((mb_dim,agent.s_dim),dtype=np.float32)
mb_sPrime = np.zeros((mb_dim,agent.s_dim),dtype=np.float32)
mb_r = np.zeros((mb_dim,1),dtype=np.float32)
mb_nt = np.zeros((mb_dim,1),dtype=np.float32)
#a = env.action_space.sample()
def get_mb(cond,mb_s,mb_sPrime,mb_r,mb_nt):
    if cond == 0:
        x = np.linspace(-4,4,30)
        y = np.linspace(4,-4,30)
        xv, yv = np.meshgrid(x,y)
        count = 0
        for xi in range(30):
            for yi in range(30):
                mb_s[count,:] = np.asarray([xv[xi,yi],yv[xi,yi]])
                count +=1

        _,a = agent.get_value(mb_s)
        a = sess.run(a)
        for j in range(mb_dim):
            sPrime,r,term = env.get_transition(mb_s[j],a[j])
            mb_sPrime[j,:] = sPrime
            mb_r[j] = r
            mb_nt[j] = not term
    elif cond == 1:
        for j in range(mb_dim):
            mb_s[j,:] = env.observation_space.sample().astype(np.float32)
        _,a = agent.get_value(mb_s)
        a = sess.run(a)
        for j in range(mb_dim):
            sPrime,r,term = env.get_transition(mb_s[j],a[j])
            mb_sPrime[j,:] = sPrime
            mb_r[j] = r
            mb_nt[j] = not term
get_mb(mb_cond,mb_s,mb_sPrime,mb_r,mb_nt)
for i in range(num_steps):
    _,values,mb_values,cur_grads,cur_loss = sess.run([agent.train_step,agent.cur_V,agent.init_value,agent.get_grads,agent.loss],
            feed_dict={agent._s:mb_s,agent._sPrime:mb_sPrime,agent._r:mb_r,agent._nt:mb_nt})
    cumgrads += np.abs(np.asarray(cur_grads)).sum()
    cumloss += cur_loss
    if i % refresh == 0:
        print('iter: ', i,'loss: ',cumloss,'grads: ',cumgrads,'time: ',time.clock()-cur_time)
        cur_time = time.clock()
        cumloss = 0
        cumgrads = 0
        '''inferred values'''
        plt.figure(1)
        plt.clf()
        axes = plt.gca()
        axes.set_xlim([-4,4])
        axes.set_ylim([-4,4])
        Xs = mb_s[:,0]
        Ys = mb_s[:,1]
        plt.scatter(Xs,Ys,s=np.log(mb_values+1)*1000,c=np.log(mb_values))
        '''database values'''
        plt.figure(2)
        plt.clf()
        axes = plt.gca()
        axes.set_xlim([-4,4])
        axes.set_ylim([-4,4])
        Xs = agent.SPrime_view[:,0]
        Ys = agent.SPrime_view[:,1]
        plt.scatter(Xs,Ys,s=np.log(values+1)*1000,c=np.log(values))
        plt.pause(.01)
    if agent.change_actions:
        get_mb(mb_cond,mb_s,mb_sPrime,mb_r,mb_nt)

