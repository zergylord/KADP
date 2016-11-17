import tensorflow as tf
from Utils.ops import *
import numpy as np
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
        #initial = tf.contrib.layers.xavier_initializer()
        initial = orthogonal_initializer()
        with tf.variable_scope(scope,reuse=tied):
            hid = linear(inp,self.hid_dim,'hid1',tf.nn.relu,init=initial)
            #hid = linear(hid,self.hid_dim,'hid2',tf.nn.relu,init=initial)
            last_hid = linear(hid,self.z_dim,'hid3',init=initial)
            #last_hid = linear(inp,self.z_dim,'hid1')
        if not tied: #only want to do this once
            with tf.variable_scope('network',reuse=True):
                self.net_weights = tf.get_variable('hid1/W')
                tf.histogram_summary('hid1',tf.get_variable('hid1/W'))
                #tf.histogram_summary('hid2',tf.get_variable('hid2/W'))
                tf.histogram_summary('hid3',tf.get_variable('hid3/W'))
        last_hid = tf.check_numerics(last_hid,'fuck net')
        return last_hid
    def embed(self,obs):
        x = obs
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
        elif rank >= 3:
            s = self.make_network(tf.reshape(x,[-1,self.s_dim]),tied=tie)
            if shape.__class__ == tf.Tensor:
                s = tf.reshape(s,tf.concat(0,[shape[:-1],(self.z_dim,)]))
            else:
                s = tf.reshape(s,shape[:-1]+(self.z_dim,))
        else:
            print('this shouldnt happen...')
        return s

    def kernel(self,o1,o2,mother='dot',minibatch=False):
        k = self.k
        x1 = self.embed(o1)
        x2 = self.embed(o2)
        shape1,rank1 = get_shape_info(x1)
        shape2,rank2 = get_shape_info(x2)
        '''
        print('shapes:',shape1,shape2)
        print('ranks:',rank1,rank2)
        '''
        if minibatch:
            ''' compare for each minibatch'''
            assert_op = tf.Assert(tf.equal(shape1[0],shape2[0]),[shape1,shape2])
            with tf.control_dependencies([assert_op]):
                if rank1 < rank2:
                    x1 = tf.expand_dims(x1,1)
                elif rank1 > rank2:
                    x2 = tf.expand_dims(x2,1)
                else:
                    x1=x1
        else:
            '''reshape to (n_actions,mb_dim,samples_per_action,s_dim)'''
            assert rank1==2, rank1
            assert rank2==3, rank2
            x1 = tf.expand_dims(tf.expand_dims(x1,1),0)
            '''x2 always has the first dim'''
            x2 = tf.expand_dims(x2,1)
        if mother == 'rbf':
            '''Gauassian'''
            x1 = tf.check_numerics(x1,'fuck x1')
            x2 = tf.check_numerics(x2,'fuck x2')
            sim = tf.exp(-tf.reduce_sum(tf.square(x2-x1),-1)/self.b)
            sim = tf.check_numerics(sim,'fuck sim')
        elif mother == 'dot':
            '''dot-product'''
            inv_mag1 = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(x1),-1,keep_dims=True),eps,float("inf")))
            inv_mag2 = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(x2),-1,keep_dims=True),eps,float("inf")))
            sim = tf.squeeze(tf.reduce_sum(x2*x1,-1,keep_dims=True)*inv_mag1*inv_mag2)
        k_sim,k_inds = tf.nn.top_k(sim,k=k,sorted=False)

        return k_sim,k_inds
    def _norm(self,inp):
        #return tf.nn.softmax(inp)
        #return inp/tf.reduce_sum(inp,-1,keep_dims=True)
        return (inp+1)/tf.reduce_sum(inp+1,-1,keep_dims=True)
    def _get_reward(self,inp,a=None):
        if a == None:
            S_ = self._S
            weights,inds = self.kernel(inp,S_)
        else:
            S_ = tf.gather(self._S,a)
            weights,inds = self.kernel(inp,S_,minibatch=True)
        normed_weights = self._norm(weights)
        if a == None:
            row_inds = self.row_offsets+inds
            R_ = tf.gather(self._R_view,row_inds)
        else:
            R_ = tf.gather(self._R,a)
        r = tf.reduce_sum(normed_weights*(R_),-1)
        return r
        
    def _get_value(self,inp):
        q_vals = self._get_q(inp) 
        if self.max_cond == 1:
            val = tf.reduce_sum(tf.nn.softmax(q_vals,dim=0)*q_vals,0)
        elif self.max_cond == 2:
            val = tf.reduce_mean(q_vals,0)
        else:
            val = tf.reduce_max(q_vals,0)
        action = tf.argmax(q_vals,0) #this is wasteful!
        #val = tf.Print(val,[q_vals[:,0],action[0]],summarize=10)
        return val,action
    def _get_q(self,inp,a=None):
        if a == None:
            S_ = self._S
            weights,inds = self.kernel(inp,S_)
        else:
            S_ = tf.gather(self._S,a)
            weights,inds = self.kernel(inp,S_,minibatch=True)
        #weights = tf.Print(weights,[tf.reduce_min(weights),tf.reduce_max(weights)])
        weights = tf.check_numerics(weights,'fuck weights')
        self.foo = weights
        normed_weights = self._norm(weights)
        normed_weights = tf.check_numerics(normed_weights,'fuck nw')
        if a == None:
            row_inds = self.row_offsets+inds
            R_ = tf.gather(self._R_view,row_inds)
            NT_ = tf.gather(self._NT_view,row_inds)
            V_ = tf.gather(self.V_view,row_inds)
        else:
            R_ = tf.gather(self._R,a)
            NT_ = tf.gather(self._NT,a)
            V_ = tf.gather(self.V,a)
        if self.oracle:
            if a == None:
                q_val = self._real_r + tf.reduce_sum(normed_weights*(NT_*self._gamma*V_),-1)
            else:
                q_val = self._real_ra + tf.reduce_sum(normed_weights*(NT_*self._gamma*V_),-1)
        else:
            q_val = tf.reduce_sum(normed_weights*(R_+NT_*self._gamma*V_),-1)
        return q_val
    def gen_data(self,env):
        for a in range(self.n_actions):
            for i in range(self.samples_per_action):
                s = env.observation_space.sample()
                sPrime,r,term = env.get_transition(s,a)
                self.S[a,i] = s
                self.SPrime[a,i] = sPrime
                self.R[a,i] = r
                self.NT[a,i] = np.float32(not term)
        for a in range(self.n_actions):
            for i in range(self.n_samples):
                #for oracle
                _,self.RPrime[a,i],_ = env.get_transition(self.SPrime_view[i],a)

    def __init__(self,env,W_and_NNI = None):
        self.net_exists = False
        self.n_actions = env.action_space.n
        self.samples_per_action = 400
        self.k = 400
        self.oracle = True
        self.n_samples = self.n_actions*self.samples_per_action
        #for converting inds for a particular action to row inds
        self.row_offsets = np.expand_dims(np.expand_dims(np.arange(self.n_actions)*self.samples_per_action,-1),-1) 
        
        self.s_dim = 2
        self.z_dim = 10
        self.b = .01
        self.hid_dim = 256
        self.lr = 1e-4
        self.max_cond = 3 #1 softmax,2 mean, 3+ max
        self.change_actions = True
        ''' all placeholders'''
        self._s = tf.placeholder(tf.float32,shape=(None,self.s_dim,))
        self._a = tf.placeholder(tf.int32,shape=(None,))
        self._r = tf.placeholder(tf.float32,shape=(None,1,))
        self._real_r = tf.placeholder(tf.float32,shape=(self.n_actions,None),name='real_r')
        self._real_ra = tf.placeholder(tf.float32,shape=(None,),name='real_ra')
        self._sPrime = tf.placeholder(tf.float32,shape=(None,self.s_dim,))
        self._nt = tf.placeholder(tf.float32,shape=(None,1,))
        self._gamma = tf.placeholder(tf.float32,shape=())

        self._S = tf.placeholder(tf.float32,shape=(self.n_actions,self.samples_per_action,self.s_dim))
        self._SPrime_view = tf.placeholder(tf.float32,shape=(self.n_samples,self.s_dim))
        self._R = tf.placeholder(tf.float32,shape=(self.n_actions,self.samples_per_action,),name='R')
        self._R_view = tf.reshape(self._R,(-1,))
        self._RPrime = tf.placeholder(tf.float32,shape=(self.n_actions,self.n_samples,),name='RPrime')
        self._NT = tf.placeholder(tf.float32,shape=(self.n_actions,self.samples_per_action,),name='NT')
        self._NT_view = tf.reshape(self._NT,(-1,))
        '''create dataset'''
        self.S = np.zeros((self.n_actions,self.samples_per_action,self.s_dim)).astype(np.float32())
        self.S_view = self.S.reshape(-1,self.s_dim)
        self.SPrime = np.zeros((self.n_actions,self.samples_per_action,self.s_dim)).astype(np.float32())
        self.SPrime_view = self.SPrime.reshape(-1,self.s_dim)
        self.R = np.zeros((self.n_actions,self.samples_per_action)).astype(np.float32())
        self.R_view = self.R.reshape(-1)
        self.RPrime = np.zeros((self.n_actions,self.n_samples)).astype(np.float32())
        self.NT = np.zeros((self.n_actions,self.samples_per_action)).astype(np.float32())
        self.NT_view = self.NT.reshape(-1)
        ''' these should be tensors
        self.V = np.zeros((self.n_actions,self.samples_per_action))
        self.V_view = self.V.reshape(-1)
        '''
        self.gen_data(env)
        ''' create similarity sparse matrix'''
        if W_and_NNI == None:
            self.W,inds = self.kernel(self._SPrime_view,self._S)
            tf.histogram_summary('W',self.W)
            self.NNI = self.row_offsets+inds
        else:
            self.W,self.NNI = W_and_NNI
        '''create computation graph'''
        normed_W = self._norm(self.W)
        self.max_prob = tf.reduce_mean(tf.reduce_max(normed_W,-1))
        V = [tf.zeros((self.n_samples,),dtype=tf.float32)]
        inds = self.NNI
        print(self.row_offsets.shape)
        R_ = tf.gather(self._R_view,inds)
        NT_ = tf.gather(self._NT_view,inds)
        for t in range(10):
            V_ = tf.gather(V[t],inds)
            if self.oracle:
                q_vals = self._RPrime+tf.reduce_sum(normed_W*(NT_*self._gamma*V_),-1)
            else:
                q_vals = tf.reduce_sum(normed_W*(R_+NT_*self._gamma*V_),-1)
            if self.max_cond == 1:
                V.append(tf.reduce_sum(tf.nn.softmax(q_vals,dim=0)*q_vals,0))
            elif self.max_cond == 2:
                V.append(tf.reduce_mean(q_vals,0))
            else:
                V.append(tf.reduce_max(q_vals,0))
        self.V_view= V[-1]
        self.val_diff = tf.reduce_sum(tf.square(V[-1]-V[-2]))
        '''
        def loop_func(V,count):
            V_ = tf.gather(V,inds)
            if self.oracle:
                q_vals = self._RPrime+tf.reduce_sum(normed_W*(NT_*self._gamma*V_),-1)
            else:
                q_vals = tf.reduce_sum(normed_W*(R_+NT_*self._gamma*V_),-1)
            if self.max_cond == 1:
                ret = tf.reduce_sum(tf.nn.softmax(q_vals,dim=0)*q_vals,0)
            elif self.max_cond == 2:
                ret = tf.reduce_mean(q_vals,0)
            else:
                ret = tf.reduce_max(q_vals,0)
            ret.set_shape((self.n_samples,))
            return (ret,count+1)
        cond = lambda V,count: count<15
        self.V_view,_ = tf.while_loop(cond,loop_func,[tf.zeros((self.n_samples,),dtype=tf.float32),tf.constant(0)])
        self.val_diff = tf.no_op()
        '''
        self.V_view = tf.check_numerics(self.V_view,'foobar')
        #self.V_view= tf.reduce_mean(tf.pack(V),0)
        self.V = tf.reshape(self.V_view,[self.n_actions,self.samples_per_action])
        '''get value graph
            feed: _s
            ops: val,action
        '''
        self.val,self.action = self._get_value(self._s)
        self.q_val = self._get_q(self._s)
        self.r_val = self._get_reward(self._s)
        '''TD graph
            feed: _s,_r,_sPrime,_nt
            ops: train_step,get_grads,loss
            NOTE: This TD error is incorrect for exploratory actions
        '''
        self.final_value,_ = self._get_value(self._sPrime)
        target = tf.stop_gradient(self._r + self._nt*self._gamma*self.final_value)
        self.v_loss = tf.reduce_mean(tf.square(target-self.val))
        self.train_v = tf.train.AdamOptimizer(self.lr).minimize(self.v_loss)
        if self.k == self.samples_per_action:
            '''Q learning graph
                feed: _s,_a,_r,_sPrime,_nt
                ops: train_q
            '''
            self.q = self._get_q(self._s,self._a)
            self.q = tf.check_numerics(self.q,'fuck q')
            self.target_val = tf.placeholder(tf.float32,[None,],name="target_val")
            self.target_q = tf.stop_gradient(self._r+self._nt*self._gamma*self.target_val)
            self.q_loss = tf.reduce_mean(tf.square(self.target_q - self.q))
            tf.scalar_summary('q loss',self.q_loss)
            self.q_loss = tf.check_numerics(self.q_loss,'fuck q loss')
            optim = tf.train.AdamOptimizer(self.lr)
            grads_and_vars = optim.compute_gradients(self.q_loss)
            capped_grads_and_vars = grads_and_vars
            #capped_grads_and_vars = [(tf.clip_by_value(gv[0],-.1,.1),gv[1]) for gv in grads_and_vars]
            grad_summaries = [tf.histogram_summary('poo'+v.name,g) if g is not None else '' for g,v in grads_and_vars]
            self.train_q = optim.apply_gradients(capped_grads_and_vars)
            '''reward and statePred training graph
                feed: _s,_a,_r,_sPrime,_nt
                ops: train_supervised
            '''
            gathered_S = tf.gather(self._S,self._a)
            gathered_SPrime = self._SPrime_view #tf.gather(self.SPrime,self._a)
            gathered_R = tf.gather(self._R,self._a)
            weights,_ = self.kernel(self._s,gathered_S,minibatch=True)
            normed_weights = self._norm(weights)
            r = tf.reduce_sum(normed_weights*gathered_R,1)
            #r = tf.Print(r,[tf.nn.zero_fraction(self._R),tf.nn.zero_fraction(gathered_R)],'hello there')
            action_W,_ = self.kernel(tf.expand_dims(tf.expand_dims(gathered_SPrime,0),0)
                    ,tf.expand_dims(gathered_S,2),minibatch=True)
            pred_s = self._norm(tf.squeeze(tf.batch_matmul(tf.expand_dims(normed_weights,1),action_W)))
            target_s = self._norm(self.kernel(self._sPrime,gathered_S,minibatch=True)[0])

            mse = tf.square(target_s-pred_s)
            #mse = tf.Print(mse,[tf.reduce_sum(target_s,1),tf.reduce_sum(pred_s,1)])
            dot = tf.square(target_s*pred_s)
            self.s_loss = tf.reduce_sum(self._nt*mse)/tf.reduce_sum(self._nt)
            self.r_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self._r-r),1))
            tf.scalar_summary('reward loss',self.r_loss)
            self.super_loss = self.r_loss #+self.s_loss
            self.train_supervised = tf.train.AdamOptimizer(self.lr).minimize(self.super_loss)
            '''
            pred_s = tf.gather(self.W[0],row_inds)
            next_inds = tf.gather(self.NNI[0],row_inds)
            '''
            '''Return based training graph (MC supervised)
                feed:_s,_a,_r AS RETURN,_nt
                op: train_return
            '''
            self.train_return = tf.train.AdamOptimizer(self.lr).minimize(self.r_loss)
        self.get_grads = tf.reduce_mean(tf.reduce_sum(tf.gradients(self.q_loss,self.net_weights),-1))
        self.zero_fraction = tf.nn.zero_fraction(self._R)

