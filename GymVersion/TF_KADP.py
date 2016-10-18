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
        dist = tf.square(x-X)
        sim = tf.exp(-tf.clip(dist,0,10))
        k_sim,k_inds = tf.nn.top_k(sim,k=self.k,sorted=False)
        return k_sim,k_inds
    def tf_get_value(self,s):
        
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
        self.samples_per_action = 1000
        self.n_samples = self.n_actions*self.samples_per_action
        
        self.s_dim = 2
        self.k = 5
        self.b = 1e-1
        self.gamma = .9
        '''create dataset'''
        self.S = np.zeros((self.n_actions,self.samples_per_action,self.s_dim))
        self.SPrime = np.zeros((self.n_actions,self.samples_per_action,self.s_dim))
        self.SPrime_view = self.SPrime.reshape(-1,self.s_dim)
        self.R = np.zeros((self.n_actions,self.samples_per_action))
        self.R_view = self.R.reshape(-1)
        self.NT = np.zeros((self.n_actions,self.samples_per_action))
        self.NT_view = self.NT.reshape(-1)
        self.V = np.zeros((self.n_actions,self.samples_per_action))
        self.V_view = self.V.reshape(-1)
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
        normed_W = tf.nn.softmax(tf_W,dim=2)
        V = [tf.zeros((self.n_samples,),dtype=tf.float64)]
        inds = self.NNI
        R_ = tf.gather(self.R_view,inds)
        NT_ = tf.gather(self.NT_view,inds)
        for t in range(100):
            V_ = tf.gather(V[t],inds)
            V.append(tf.reduce_max(tf.reduce_sum(normed_W*(R_+NT_*self.gamma*V_),-1),0))
        self._target = tf.placeholder(tf.float64,shape=(self.n_samples,))
        self.cur_V = V[-1]
        '''NOTE: This TD error is incorrect for exploratory actions'''
        self._s = tf.placeholder(tf.float64,shape=(self.s_dim,))
        self._r = tf.placeholder(tf.float64,shape=(1,))
        self._sPrime = tf.placeholder(tf.float64,shape=(self.s_dim,))
        init_value = self.get_value(self._s)
        final_value = self.get_value(self._sPrime)
        target = tf.stop_gradient(self._r + self.gamma*final_value)
        loss = tf.reduce_sum(tf.square(target-init_value))
        self.get_grads = tf.gradients(loss,tf_W)
        self.train_step = tf.train.AdamOptimizer().minimize(loss)
env = simple_env.Simple()
agent = KADP(env)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
cur_time = time.clock()
for i in range(10):
    s = env.observation_space.sample()
    _,a = agent.get_value(s)
    sPrime = env.get_transition(s,a)
    r = env.get_reward(sPrime)
    [_,bar,foo] = sess.run([agent.train_step,agent.get_grads,agent.cur_V],feed_dict={agent._s:s,agent._sPrime:sPrime,agent._r:r})
    print(bar,foo,time.clock()-cur_time)
    plt.figure(1)
    plt.clf()
    axes = plt.gca()
    axes.set_xlim([-4,4])
    axes.set_ylim([-4,4])
    plt.scatter(agent.SPrime_view[:,0],agent.SPrime_view[:,1],s=np.log(foo+1)*100,c=np.log(foo))
    plt.figure(2)
    plt.clf()
    plt.hist(bar[0].flatten())
    plt.pause(.01)
