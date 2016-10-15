import time
import numpy as np
import tensorflow as tf
import simple_env
class KADP(object):
    def __init__(self,env):
        self.n_actions = env.action_space.n
        self.samples_per_action = 100
        self.n_samples = self.n_actions*self.samples_per_action
        
        self.s_dim = 2
        self.k = 5
        self.b = 1e-1
        self.gamma = .9
        '''create dataset'''
        self.S = np.zeros((self.n_actions,self.samples_per_action,self.s_dim))
        self.SPrime = np.zeros((self.n_actions,self.samples_per_action,self.s_dim))
        self.R = np.zeros((self.n_actions,self.samples_per_action))
        self.NT = np.zeros((self.n_actions,self.samples_per_action))
        self.V = np.zeros((self.n_actions,self.samples_per_action))
        for a in range(self.n_actions):
            for i in range(self.samples_per_action):
                s = env.observation_space.sample()
                sPrime,r,term = env.get_transition(s,a)
                self.S[a,i] = s
                self.SPrime[a,i] = sPrime
                self.R[a,i] = r
                self.NT[a,i] = np.float32(not term)
        '''create computation graph'''


W = tf.random_uniform((100,5))
normed_W = tf.nn.softmax(W)
R = tf.random_uniform((100,))
inds = np.random.choice(100,5*100).reshape(100,5)

R_ = tf.gather(R,inds)
V = [tf.zeros((100,))]
#V_ = tf.gather(V[0],inds)
for t in range(1000):
    V_ = tf.gather(V[t],inds)
    V.append(tf.reduce_sum(normed_W*(R_+.9*V_),1))
_target = tf.placeholder(tf.float32,shape=(100,))
loss = tf.reduce_sum(tf.square(_target-V[-1]))
get_grads = tf.gradients(loss,W)
sess = tf.Session()
sess.run(tf.initialize_all_variables())
cur_time = time.clock()
for i in range(100):
    foo = sess.run(get_grads,feed_dict={_target:np.random.randn(100)})
print(foo,time.clock()-cur_time)

env = simple_env.Simple()
