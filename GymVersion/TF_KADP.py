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
def get_shape_info(x):
    if x.__class__ == tf.Tensor:
        shape = tf.shape(x)
        rank = len(x.get_shape())
    else:
        shape = x.shape
        rank = len(shape)
    return shape,rank

class KADP(object):
    def make_network(self,inp,scope='network',tied=False):
        with tf.variable_scope(scope,reuse=tied):
            hid = linear(inp,self.hid_dim,'hid1',tf.nn.relu)
            #hid = linear(hid,self.hid_dim,'hid2',tf.nn.relu)
            last_hid = linear(hid,self.s_dim,'hid3')
        return last_hid
    def embed(self,x):
        if not self.net_exists:
            print('new network!')
            tie = False
            self.net_exists = True
        else:
            tie = True
        shape,rank = get_shape_info(x)
        if rank == 1:
            s = self.make_network(tf.expand_dims(x,0),tied=tie)
        elif rank == 2:
            s = self.make_network(x,tied=tie)
        elif rank == 3:
            s = self.make_network(tf.reshape(x,[-1,self.s_dim]),tied=tie)
            s = tf.reshape(s,shape)
        else:
            print('this shouldnt happen...')
        return s

    def kernel(self,x1,x2,k=None):
        if k == None:
            k = self.k
        x1 = self.embed(x1)
        x2 = self.embed(x2)
        shape1,rank1 = get_shape_info(x1)
        shape2,rank2 = get_shape_info(x2)
        if rank2 == rank2 and rank2 == 3:
            print('poop')
            ''' compare all combination'''
            x1 = tf.expand_dims(x1,1)
            x2 = tf.expand_dims(x2,2)
        else:
            '''reshape to (n_actions,mb_dim,samples_per_action,s_dim)'''
            if rank1 == 2:
                x1 = tf.expand_dims(tf.expand_dims(x1,1),0)
            else:
                print('this shouldnt happen...')
            '''either (n_samples,s_dim), or (a_dim,samples_per_action,s_dim)'''
            if rank2 == 2:
                x2 = tf.expand_dims(tf.expand_dims(x2,0),0)
            elif rank2 == 3:
                x2 = tf.expand_dims(x2,1)
            else:
                print('this shouldnt happen...')
        '''Gaussian'''
        '''
        dist = tf.sqrt(tf.reduce_sum(tf.square(x1-x2),-1))
        sim = tf.exp(-tf.clip_by_value(dist,0,10))
        '''
        '''dot-product'''
        inv_mag = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(x2),-1,keep_dims=True),eps,float("inf")))
        sim = tf.squeeze(tf.reduce_sum(x2*x1,-1,keep_dims=True)*inv_mag)
        k_sim,k_inds = tf.nn.top_k(sim,k=k,sorted=False)

        return k_sim,k_inds
    def _get_value(self,inp):
        weights,inds = self.kernel(inp,self.S)
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
        self.k = 100
        self.b = 1e1
        self.hid_dim = 64
        self.lr = 1e-2
        self.softmax = True
        self.change_actions = False
        self.gamma = .9
        '''create dataset'''
        self.S = np.zeros((self.n_actions,self.samples_per_action,self.s_dim)).astype(np.float32())
        self.S_view = self.S.reshape(-1,self.s_dim)
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
        ''' all placeholders'''
        self._s = tf.placeholder(tf.float32,shape=(None,self.s_dim,))
        self._a = tf.placeholder(tf.int32,shape=(None,))
        self._r = tf.placeholder(tf.float32,shape=(None,1,))
        self._sPrime = tf.placeholder(tf.float32,shape=(None,self.s_dim,))
        self._nt = tf.placeholder(tf.float32,shape=(None,1,))
        '''TD graph
            feed: _s,_r,_sPrime,_nt,init_value,final_value
            ops: train_step,get_grads,loss
            NOTE: This TD error is incorrect for exploratory actions
        '''
        self.init_value,_ = self._get_value(self._s)
        self.final_value,_ = self._get_value(self._sPrime)
        target = tf.stop_gradient(self._r + self._nt*self.gamma*self.final_value)
        self.loss = tf.reduce_mean(tf.square(target-self.init_value))
        with tf.variable_scope('network/hid1',reuse=True):
            net_weights = tf.get_variable('W')
        self.get_grads = tf.gradients(self.loss,net_weights)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        '''get value graph
            feed: _s
            ops: val,action
        '''
        self.val,self.action = self._get_value(self._s)
        '''reward and statePred training graph
            feed: _s,_a,_r,_sPrime,_nt
            ops: train_supervised
        '''
        ''' this ignored actions...

        weights,_ = self.kernel(self._s,self.S_view,k=self.n_samples)
        normed_weights = tf.nn.softmax(weights)
        print(normed_weights.get_shape(),self.R_view.shape)
        r = tf.squeeze(tf.matmul(normed_weights,np.expand_dims(self.R_view,1)),[1])
        W_view,_ = self.kernel(self.SPrime_view,self.S_view,k=self.n_samples)
        next_weights = tf.matmul(normed_weights,W_view)
        #next_weights = tf.matmul(normed_weights,tf.reshape(self.W,[self.n_samples,self.n_samples])) #is this reshaping properly? no.
        target_s,_ = self.kernel(self._sPrime,self.S_view,k=self.n_samples)
        '''
        gathered_S = tf.gather(self.S,self._a)
        gathered_SPrime = tf.gather(self.SPrime,self._a)
        gathered_R = tf.gather(self.R,self._a)
        weights,_ = self.kernel(self._s,gathered_S)
        normed_weights = tf.nn.softmax(weights)
        r = tf.reduce_sum(normed_weights*gathered_R,1)
        action_W,_ = self.kernel(gathered_S,gathered_SPrime)
        action_W = tf.Print(action_W,[tf.shape(action_W),tf.shape(normed_weights)],'meowmeow')
        next_weights = tf.squeeze(tf.batch_matmul(tf.expand_dims(normed_weights,1),action_W))
        target_s,_ = self.kernel(self._sPrime,gathered_S)

        self.s_loss = tf.reduce_mean(tf.reduce_sum(self._nt*tf.square(target_s-next_weights),1))
        self.r_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self._r-r),1))/10
        self.super_loss = self.s_loss + self.r_loss
        self.train_supervised = tf.train.AdamOptimizer(self.lr).minimize(self.super_loss)
        '''
        next_weights = tf.gather(self.W[0],row_inds)
        next_inds = tf.gather(self.NNI[0],row_inds)
        '''
        #TODO: scatter weights to full length vector
        #TODO: concatenate across all possible actions
env = simple_env.Simple()
agent = KADP(env)
sess.run(tf.initialize_all_variables())
cur_time = time.clock()
epsilon = .1

cumloss = 0
cumgrads = 0
num_steps = int(1e6)
refresh = int(1e3)
mb_cond = 0
mb_dim = 900
mb_s = np.zeros((mb_dim,agent.s_dim),dtype=np.float32)
mb_a = np.zeros((mb_dim,),dtype=np.int32)
mb_sPrime = np.zeros((mb_dim,agent.s_dim),dtype=np.float32)
mb_r = np.zeros((mb_dim,1),dtype=np.float32)
mb_nt = np.zeros((mb_dim,1),dtype=np.float32)
#a = env.action_space.sample()
def get_mb(cond,mb_s,mb_a,mb_r,mb_sPrime,mb_nt):
    if cond == 0:
        x = np.linspace(-4,4,30)
        y = np.linspace(4,-4,30)
        xv, yv = np.meshgrid(x,y)
        count = 0
        for xi in range(30):
            for yi in range(30):
                mb_s[count,:] = np.asarray([xv[xi,yi],yv[xi,yi]])
                count +=1

        mb_a = sess.run(agent.action,feed_dict={agent._s:mb_s})
        for j in range(mb_dim):
            sPrime,r,term = env.get_transition(mb_s[j],mb_a[j])
            mb_sPrime[j,:] = sPrime
            mb_r[j] = r
            mb_nt[j] = not term
    elif cond == 1:
        for j in range(mb_dim):
            mb_s[j,:] = env.observation_space.sample().astype(np.float32)
        mb_a = sess.run(agent.action,feed_dict={agent._s:mb_s})
        for j in range(mb_dim):
            sPrime,r,term = env.get_transition(mb_s[j],mb_a[j])
            mb_sPrime[j,:] = sPrime
            mb_r[j] = r
            mb_nt[j] = not term
get_mb(mb_cond,mb_s,mb_a,mb_r,mb_sPrime,mb_nt)
for i in range(num_steps):
    _,values,mb_values,cur_grads,cur_loss = sess.run([agent.train_supervised,agent.cur_V,agent.init_value,agent.get_grads,agent.super_loss],
            feed_dict={agent._s:mb_s,agent._a:mb_a,agent._sPrime:mb_sPrime,agent._r:mb_r,agent._nt:mb_nt})
    cumgrads += np.abs(np.asarray(cur_grads)).sum()
    cumloss += cur_loss
    if i % refresh == 0:
        print('iter: ', i,'loss: ',cumloss/refresh,'grads: ',cumgrads/refresh,'time: ',time.clock()-cur_time)
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

