import time
import numpy as np
import tensorflow as tf
import simple_env
from matplotlib import pyplot as plt
plt.ion()
from scipy.spatial.distance import cdist,pdist
class KADP(object):
    avg_sim = 0.0
    #TODO:make kernel and get_value all tf
    def kernel(self,x,X):
        '''Gaussian'''
        dist = np.squeeze(cdist(np.expand_dims(x,0),X)/self.b)
        sim = np.exp(-(np.clip(dist,0,10)))
        inds = np.argpartition(dist,self.k-1)[:self.k]
        self.avg_sim = self.avg_sim*.99+sim.sum()*.01
        assert np.all(sim>0), sim[sim<=0]
        return inds,sim
    def tf_kernel(self,x,X):
        dist = tf.sqrt(tf.reduce_sum(tf.square(x-X),-1))
        sim = tf.exp(-tf.clip_by_value(dist,0,10))
        k_sim,k_inds = tf.nn.top_k(sim,k=self.k,sorted=False)
        return k_sim,k_inds
    def tf_get_value(self,s):
        weights,inds = self.tf_kernel(s,self.S)
        normed_weights = tf.nn.softmax(weights)
        row_inds = self.row_offsets+inds
        R_ = tf.gather(self.R_view,row_inds)
        NT_ = tf.gather(self.NT_view,row_inds)
        V_ = tf.gather(self.cur_V,row_inds)
        q_vals = tf.reduce_sum(normed_weights*(R_+NT_*self.gamma*V_),-1)
        val = tf.reduce_max(q_vals,0)
        action = tf.argmax(q_vals,0) #this is wasteful!
        return val,action

        
    ''' calc current value of single state s '''
    def get_value(self,s):
        cur_V = np.zeros((self.n_actions,))
        for a in range(self.n_actions):
            weights = np.zeros((self.samples_per_action,))
            inds,vals = self.kernel(s,self.S[a])
            weights[inds] = vals[inds]
            cur_W_a = tf.nn.softmax(weights)
            cur_V[a] = cur_W_a.dot(self.R[a]+self.gamma*self.V[a])
            aind = cur_V.argmax(0)
        return cur_V[aind],aind
    def __init__(self,env,W_and_NNI = None):
        self.n_actions = env.action_space.n
        self.samples_per_action = 100
        self.n_samples = self.n_actions*self.samples_per_action
        #for converting inds for a particular action to row inds
        self.row_offsets = np.expand_dims(np.arange(self.n_actions)*self.samples_per_action,-1) 
        
        self.s_dim = 2
        self.k = 5
        self.b = 1e1
        self.gamma = .9
        '''create dataset'''
        self.S = np.zeros((self.n_actions,self.samples_per_action,self.s_dim))
        self.SPrime = np.zeros((self.n_actions,self.samples_per_action,self.s_dim))
        self.SPrime_view = self.SPrime.reshape(-1,self.s_dim)
        self.R = np.zeros((self.n_actions,self.samples_per_action))
        self.R_view = self.R.reshape(-1)
        self.NT = np.zeros((self.n_actions,self.samples_per_action))
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
                self.NT[a,i] = np.float64(not term)
        ''' create similarity sparse matrix'''
        if W_and_NNI == None:
            self.W = np.zeros((self.n_actions,self.n_samples,self.k))
            self.NNI = np.zeros((self.n_actions,self.n_samples,self.k)).astype(np.int)
            for i in range(self.n_samples):
                for a in range(self.n_actions):
                    inds,sim = self.kernel(self.SPrime_view[i],self.S[a])
                    self.W[a,i,:] = sim[inds]
                    row_inds = a*self.samples_per_action+inds
                    self.NNI[a,i,:] = row_inds
        else:
            self.W,self.NNI = W_and_NNI
        '''create computation graph'''
        tf_W = tf.Variable(self.W)
        normed_W = tf.nn.softmax(tf_W)
        V = [tf.zeros((self.n_samples,),dtype=tf.float64)]
        self.cur_V = V[-1]
        inds = self.NNI
        R_ = tf.gather(self.R_view,inds)
        NT_ = tf.gather(self.NT_view,inds)
        for t in range(100):
            V_ = tf.gather(V[t],inds)
            q_vals = tf.reduce_sum(normed_W*(R_+NT_*self.gamma*V_),-1)
            V.append(tf.reduce_max(q_vals,0))
            self.cur_V = V[-1]
            foo,_ = self.tf_get_value(self.SPrime_view[0])
            tf.Assert(foo == V[-1][0],[foo,V[-1][0]])
        '''NOTE: This TD error is incorrect for exploratory actions'''
        self._s = tf.placeholder(tf.float64,shape=(self.s_dim,))
        self._r = tf.placeholder(tf.float64,shape=(1,))
        self._sPrime = tf.placeholder(tf.float64,shape=(self.s_dim,))
        init_value,_ = self.tf_get_value(self._s)
        final_value,_ = self.tf_get_value(self._sPrime)
        target = tf.stop_gradient(self._r + self.gamma*final_value)
        self.loss = tf.reduce_sum(tf.square(target-init_value))
        self.get_grads = tf.gradients(self.loss,tf_W)
        self.train_step = tf.train.AdamOptimizer().minimize(self.loss)
env = simple_env.Simple()
agent = KADP(env)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
cur_time = time.clock()
epsilon = .1

cumloss = 0
for i in range(100):
    s = env.observation_space.sample()
    '''
    if np.random.rand() < epsilon:
        a = env.action_space.sample()
    else:
    '''
    _,a = agent.tf_get_value(s)
    a = sess.run(a)
    sPrime,r,term = env.get_transition(s,a)
    _,values,grads,cur_loss = sess.run([agent.train_step,agent.cur_V,agent.get_grads,agent.loss],feed_dict={agent._s:s,agent._sPrime:sPrime,agent._r:[r]})
    cumloss += cur_loss
    if i % 10 == 0:
        print('iter: ', i,'loss: ',cumloss,'grads: ',np.abs(np.asarray(grads)).sum(),'time: ',time.clock()-cur_time)
        cur_time = time.clock()
        cumloss = 0

    plt.figure(1)
    plt.clf()
    axes = plt.gca()
    axes.set_xlim([-4,4])
    axes.set_ylim([-4,4])
    plt.scatter(agent.SPrime_view[:,0],agent.SPrime_view[:,1],s=np.log(values+1)*100,c=np.log(values))
    '''
    plt.figure(2)
    plt.clf()
    grads = grads[0].flatten()
    plt.hist(grads[grads!=0])
    '''
    plt.pause(.01)
