import time
import numpy as np
import tensorflow as tf
np.random.seed(111)
#tf.set_random_seed(111)
print(np.random.rand())
sess = tf.Session()
foo = sess.run(tf.random_uniform((1,)))
print('hi',foo)
import simple_env
from ops import *
from matplotlib import pyplot as plt
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
        initial = tf.contrib.layers.xavier_initializer()
        #initial = orthogonal_initializer()
        with tf.variable_scope(scope,reuse=tied):
            #hid = linear(inp,self.hid_dim,'hid1',tf.nn.relu,init=initial)
            #hid = linear(hid,self.hid_dim,'hid2',tf.nn.relu)
            #last_hid = linear(hid,self.z_dim,'hid3',init=initial)
            last_hid = linear(inp,self.z_dim,'hid1')
        if not tied: #only want to do this once
            with tf.variable_scope('network',reuse=True):
                self.net_weights = tf.get_variable('hid1/W')
                tf.histogram_summary('hid1',tf.get_variable('hid1/W'))
                '''
                tf.histogram_summary('hid2',tf.get_variable('hid2/W'))
                tf.histogram_summary('hid3',tf.get_variable('hid3/W'))
                '''
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

    def kernel(self,o1,o2,k=None,minibatch=False):
        if k == None:
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
        '''Gauassian'''
        x1 = tf.check_numerics(x1,'fuck x1')
        x2 = tf.check_numerics(x2,'fuck x2')
        sim = tf.exp(-tf.reduce_sum(tf.square(x2-x1),-1)/self.b)
        sim = tf.check_numerics(sim,'fuck sim')
        '''dot-product'''
        '''
        inv_mag = tf.rsqrt(tf.clip_by_value(tf.reduce_sum(tf.square(x2),-1,keep_dims=True),eps,float("inf")))
        sim = tf.squeeze(tf.reduce_sum(x2*x1,-1,keep_dims=True)*inv_mag)
        '''
        k_sim,k_inds = tf.nn.top_k(sim,k=k,sorted=False)

        return k_sim,k_inds
    def _norm(self,inp):
        #return tf.nn.softmax(inp)
        return inp/tf.reduce_sum(inp,-1,keep_dims=True)
    def _get_value(self,inp):
        q_vals = self._get_q(inp) 
        if self.softmax:
            val = tf.reduce_sum(tf.nn.softmax(q_vals,dim=0)*q_vals,0)
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

    def __init__(self,env,W_and_NNI = None):
        self.net_exists = False
        self.n_actions = env.action_space.n
        self.samples_per_action = 100
        self.k = 100
        self.n_samples = self.n_actions*self.samples_per_action
        #for converting inds for a particular action to row inds
        self.row_offsets = np.expand_dims(np.expand_dims(np.arange(self.n_actions)*self.samples_per_action,-1),-1) 
        
        self.s_dim = 2
        self.z_dim = 2
        self.b = 1
        self.hid_dim = 64
        self.lr = 1e-4
        self.softmax = False
        self.change_actions = True
        ''' all placeholders'''
        self._s = tf.placeholder(tf.float32,shape=(None,self.s_dim,))
        self._a = tf.placeholder(tf.int32,shape=(None,))
        self._r = tf.placeholder(tf.float32,shape=(None,1,))
        self._sPrime = tf.placeholder(tf.float32,shape=(None,self.s_dim,))
        self._nt = tf.placeholder(tf.float32,shape=(None,1,))
        self._gamma = tf.placeholder(tf.float32,shape=())

        self._S = tf.placeholder(tf.float32,shape=(self.n_actions,self.samples_per_action,self.s_dim))
        self._SPrime_view = tf.placeholder(tf.float32,shape=(self.n_samples,self.s_dim))
        self._R = tf.placeholder(tf.float32,shape=(self.n_actions,self.samples_per_action,))
        self._R_view = tf.reshape(self._R,(-1,))
        self._NT = tf.placeholder(tf.float32,shape=(self.n_actions,self.samples_per_action,))
        self._NT_view = tf.reshape(self._NT,(-1,))
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
        R_ = tf.gather(self._R_view,inds)
        NT_ = tf.gather(self._NT_view,inds)
        for t in range(32):
            V_ = tf.gather(V[t],inds)
            q_vals = tf.reduce_sum(normed_W*(R_+NT_*self._gamma*V_),-1)
            if self.softmax:
                V.append(tf.reduce_sum(tf.nn.softmax(q_vals,dim=0)*q_vals,0))
            else:
                V.append(tf.reduce_max(q_vals,0))
        #V = tf.Print(V,[V[-1]],'poo')
        self.V_view= V[-1]
        self.val_diff = tf.reduce_sum(tf.square(V[-1]-V[-2]))
        self.V_view = tf.check_numerics(self.V_view,'foobar')
        #self.V_view= tf.reduce_mean(tf.pack(V),0)
        self.V = tf.reshape(self.V_view,[self.n_actions,self.samples_per_action])
        '''get value graph
            feed: _s
            ops: val,action
        '''
        self.val,self.action = self._get_value(self._s)
        self.q_val = self._get_q(self._s)
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
            self.target_q = tf.stop_gradient(self._r+self._nt*self._gamma*self._get_value(self._sPrime)[0])
            self.q_loss = tf.reduce_mean(tf.square(self.target_q - self.q))
            self.q_loss = tf.check_numerics(self.q_loss,'fuck q loss')
            optim = tf.train.AdamOptimizer(self.lr)
            grads_and_vars = optim.compute_gradients(self.q_loss)
            capped_grads_and_vars = [(tf.clip_by_value(gv[0],-100,100),gv[1]) for gv in grads_and_vars]
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
            self.super_loss = self.r_loss #+self.s_loss
            self.train_supervised = tf.train.AdamOptimizer(self.lr).minimize(self.super_loss)
            '''
            pred_s = tf.gather(self.W[0],row_inds)
            next_inds = tf.gather(self.NNI[0],row_inds)
            '''
        self.get_grads = tf.reduce_mean(tf.reduce_sum(tf.gradients(self.q_loss,self.net_weights),-1))
        self.zero_fraction = tf.nn.zero_fraction(self._R)
env = simple_env.Simple(3)
agent = KADP(env)
check_op = tf.add_check_numerics_ops() 
merged = tf.merge_all_summaries()
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summary_dir', '/tmp/kadp', 'Summaries directory')
if tf.gfile.Exists(FLAGS.summary_dir):
    tf.gfile.DeleteRecursively(FLAGS.summary_dir)
    tf.gfile.MakeDirs(FLAGS.summary_dir)
train_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/train',sess.graph)
sess.run(tf.initialize_all_variables())
cur_time = time.clock()
epsilon = 1.0

cumloss = 0
cumgrads = 0
num_steps = int(1e8)
refresh = int(1e2)
mb_cond = 1
if mb_cond == 0:
    mb_dim = 100
else:
    mb_dim = 100
mb_s = np.zeros((mb_dim,agent.s_dim),dtype=np.float32)
mb_a = np.zeros((mb_dim,),dtype=np.int32)
mb_sPrime = np.zeros((mb_dim,agent.s_dim),dtype=np.float32)
mb_r = np.zeros((mb_dim,1),dtype=np.float32)
mb_nt = np.zeros((mb_dim,1),dtype=np.float32)
#a = env.action_space.sample()
cur_gamma = 0.0
def get_mb(cond,mb_s,mb_a,mb_r,mb_sPrime,mb_nt):
    if cond == 0:
        side = int(np.sqrt(mb_dim))
        x = np.linspace(-env.limit,env.limit,side)
        y = np.linspace(env.limit,-env.limit,side)
        xv, yv = np.meshgrid(x,y)
        count = 0
        for xi in range(side):
            for yi in range(side):
                mb_s[count,:] = np.asarray([xv[xi,yi],yv[xi,yi]])
                count +=1
        mb_s[:] = simple_env.decode(mb_s)
        mb_a = sess.run(agent.action,feed_dict={agent._R:agent.R,agent._NT:agent.NT,agent._S:agent.S,agent._SPrime_view:agent.SPrime_view,agent._gamma:cur_gamma,agent._s:mb_s})
        for j in range(mb_dim):
            sPrime,r,term = env.get_transition(mb_s[j],mb_a[j])
            mb_sPrime[j,:] = sPrime
            mb_r[j] = r
            mb_nt[j] = not term
    elif cond == 1:
        for j in range(mb_dim):
            mb_s[j,:] = env.observation_space.sample().astype(np.float32)
        mb_a = sess.run(agent.action,feed_dict={agent._R:agent.R,agent._NT:agent.NT,agent._S:agent.S,agent._SPrime_view:agent.SPrime_view,agent._gamma:cur_gamma,agent._s:mb_s})
        for j in range(mb_dim):
            sPrime,r,term = env.get_transition(mb_s[j],mb_a[j])
            mb_sPrime[j,:] = sPrime
            mb_r[j] = r
            mb_nt[j] = not term
    elif cond == 2:
        mb_s[0,:] = env.reset()
        for j in range(mb_dim):
            if j > 0:
                if term:
                    mb_s[j,:] = env.reset()
                else:
                    mb_s[j,:] = sPrime
            if np.random.rand() < epsilon:
                mb_a[j] = np.random.randint(agent.n_actions)
            else:
                mb_a[j] = sess.run(agent.action,feed_dict={agent._R:agent.R,agent._NT:agent.NT,agent._S:agent.S,agent._SPrime_view:agent.SPrime_view,agent._gamma:cur_gamma,agent._s:np.expand_dims(mb_s[j],0)})[0]
            sPrime,r,term,_ = env.step(mb_a[j])
            mb_sPrime[j,:] = sPrime
            mb_r[j] = r
            mb_nt[j] = not term

get_mb(mb_cond,mb_s,mb_a,mb_r,mb_sPrime,mb_nt)
plt.ion()
max_gamma = .9
gamma_anneal = 0 #int(1e4)
if gamma_anneal > 0:
    gamma = np.linspace(0,max_gamma,gamma_anneal).astype(np.float32)
cumr = 0
cumprob = 0
train = False
def softmax(x,dim=-1):
    ex = np.exp(x)
    denom = np.expand_dims(np.sum(ex,dim),dim)
    return ex/denom
for i in range(num_steps):
    if i < gamma_anneal:
        cur_gamma =gamma[i]
    else:
        cur_gamma = max_gamma
    if train:
        summary,_,cur_grads,cur_loss,max_prob = sess.run([merged,agent.train_q,agent.get_grads,agent.q_loss,agent.max_prob],
                feed_dict={agent._R:agent.R,agent._NT:agent.NT,agent._S:agent.S,agent._SPrime_view:agent.SPrime_view,agent._gamma:cur_gamma,agent._s:mb_s,agent._a:mb_a,agent._sPrime:mb_sPrime,agent._r:mb_r,agent._nt:mb_nt})
        train_writer.add_summary(summary)
        cumprob += max_prob
        cumgrads += cur_grads
        cumloss += cur_loss
    else:
        cumprob += 0
        cumgrads += 0
        cumloss += 0
    if i % refresh == 0:
        mb_q_values,mb_values,mb_actions,values,val_diff,embed,mb_embed,zero_frac = sess.run([agent.q_val,agent.val,agent.action,agent.V_view,agent.val_diff,agent.embed(agent.SPrime_view),agent.embed(mb_s),agent.zero_fraction]
                ,feed_dict={agent._R:agent.R,agent._NT:agent.NT,agent._S:agent.S,agent._SPrime_view:agent.SPrime_view,agent._gamma:cur_gamma,agent._s:mb_s}) 
        '''inferred values'''
        plt.figure(1)
        plt.clf()
        axes = plt.gca()
        axes.set_xlim([-env.limit,env.limit])
        axes.set_ylim([-env.limit,env.limit])
        mb_latent = simple_env.encode(mb_s)
        Xs = mb_latent[:,0]
        Ys = mb_latent[:,1]
        offX = .5*env.radius*np.cos(env.rad_inc*np.arange(agent.n_actions))
        offY = .5*env.radius*np.sin(env.rad_inc*np.arange(agent.n_actions))
        plt.hold(True)
        bub_size = 100
        if agent.softmax:
            val = np.sum(softmax(mb_q_values,0)*mb_q_values,0)
            assert np.all(np.abs(val -  mb_values) <1e-6), print(val-mb_values,np.concatenate([[val],[mb_values]]))
        else:
            assert np.all(np.max(mb_q_values,0) == mb_values)
        assert np.all(np.argmax(mb_q_values,0) == mb_actions)
        print('net reward stats: ',np.sum(agent.R,1),' mb value stats: ',np.sum(mb_q_values,1),'mb action stats: ',np.histogram(mb_actions,np.arange(agent.n_actions+1))[0])
        for action in range(agent.n_actions):
            mask = np.argmax(mb_q_values,0) == action
            #plt.scatter(Xs+offX[action],Ys+offY[action],s=bub_size*mask/2+10)#,c=((mb_q_values[action]-mb_values)))
            plt.scatter(Xs[mask]+offX[action],Ys[mask]+offY[action],s=bub_size/2)
        plt.scatter(Xs,Ys,s=bub_size,c=np.log(mb_values))
        plt.hold(False)
        '''database values'''
        plt.figure(2)
        plt.clf()
        axes = plt.gca()
        axes.set_xlim([-env.limit,env.limit])
        axes.set_ylim([-env.limit,env.limit])
        mem_latent = simple_env.encode(agent.SPrime_view)
        Xs = mem_latent[:,0]
        Ys = mem_latent[:,1]
        plt.scatter(Xs,Ys,s=100,c=np.log(values))
        if agent.z_dim == 2:
            '''model's viewpoint'''
            '''
            mb_values,mb_embed = sess.run([agent.val,agent.embed(mb_sPrime)]
                    ,feed_dict={agent._R:agent.R,agent._NT:agent.NT,agent._S:agent.S,agent._SPrime_view:agent.SPrime_view,agent._gamma:cur_gamma,agent._s:mb_sPrime}) 
            '''
            plt.figure(3)
            plt.clf()
            plt.scatter(mb_embed[:,0],mb_embed[:,1],s=bub_size,c=np.log(mb_values))
            plt.figure(4)
            plt.clf()
            plt.scatter(embed[:,0],embed[:,1],s=bub_size,c=np.log(values))
        plt.pause(.01)
        '''test performance'''
        get_mb(2,mb_s,mb_a,mb_r,mb_sPrime,mb_nt)
        print(val_diff,cumprob/refresh,zero_frac,cur_gamma,1/(mb_r.sum()/mb_dim+1e-10),'iter: ', i,'loss: ',cumloss/refresh,'grads: ',cumgrads/refresh,'time: ',time.clock()-cur_time)
        cumr = 0
        cumprob = 0
        cur_time = time.clock()
        cumloss = 0
        cumgrads = 0
        '''testing'''
        agent.gen_data(env)

    if agent.change_actions:
        get_mb(mb_cond,mb_s,mb_a,mb_r,mb_sPrime,mb_nt)
        if mb_cond == 2:
            cumr += mb_r.sum()

