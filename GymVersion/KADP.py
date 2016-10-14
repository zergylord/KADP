from matplotlib import pyplot as plt
plt.ion()
import time
import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist,pdist
from scipy.sparse import lil_matrix as sparse_matrix
from sklearn.preprocessing import normalize
import simple_env

def my_normalize(W):
    if len(W.shape) == 1:
        W = W.reshape(1,-1)
    return normalize(W,norm='l1',axis=1)
def bellman_op(W,R,V,gamma,not_term):
    return W.dot(R+not_term*gamma*V)
class KADP(object):
    def __init__(self,env):
        self.sample_ticks = int(1e3)
        self.update_ticks = int(1e2)

        self.n_actions = env.action_space.n
        self.samples_per_action = 25000
        self.n_samples = self.n_actions*self.samples_per_action
        
        self.s_dim = 64
        self.k = 5
        self.b = 1e-1
        self.gamma = .99
        self.num_buffer_obs = 1
        self.obs_dim = np.prod(env.observation_space.shape)
        self.obs_buffer = np.zeros((self.num_buffer_obs,self.obs_dim))
        self.obs_dim *= self.num_buffer_obs
        if self.obs_dim < self.s_dim:
            print('tiny state space, no projection.')
            self.s_dim = self.obs_dim
            self.transform = lambda x: self.use_obs_buffer(x)
        else:
            print('huge state space, random projection.')
            self.M = np.random.randn(self.obs_dim,self.s_dim) 
            self.transform = lambda x: np.dot(self.use_obs_buffer(x.flatten()),self.M)

        self.warming = True
        self.epsilon = 1
        ''' storage vars'''
        self.creation_time = np.asarray(np.tile(range(-self.samples_per_action,0),[self.n_actions,1]))
        self.S = np.zeros((self.n_actions,self.samples_per_action,self.s_dim))
        self.SPrime = np.zeros((self.n_actions,self.samples_per_action,self.s_dim))
        self.R = np.zeros((self.n_actions,self.samples_per_action))
        self.NT = np.zeros((self.n_actions,self.samples_per_action))
        self.V = np.zeros((self.n_actions,self.samples_per_action))
        self.WN_inds = -1*np.ones((self.n_actions,self.n_samples)).astype('int')
        self.WN_vals = np.zeros((self.n_actions,self.n_samples))
        self.mem_count = np.zeros((self.n_actions,)).astype('int')
        self.S_view = self.S.reshape(-1,self.s_dim)
        self.R_view = self.R.reshape(-1)
        self.SPrime_view = self.SPrime.reshape(-1,self.s_dim)
        self.V_view = self.V.reshape(-1)
        self.W = []
        for act in range(self.n_actions):
            self.W.append(sparse_matrix((self.n_samples,self.samples_per_action),dtype='float64'))
        ''' inds for view variables
        for when memories arent yet full '''
        self.valid_inds = []
        self.valid_mask = np.zeros((self.n_samples,),dtype='bool')

    ''' adds obs to obs history, and returns buffer '''
    buff_ptr = 0
    def use_obs_buffer(self,obs):
        self.obs_buffer[self.buff_ptr] = obs.copy()
        ret = np.roll(self.obs_buffer,-self.buff_ptr,axis=0)
        self.buff_ptr  = (self.buff_ptr-1) % self.num_buffer_obs
       # print('first: ',obs,'all: ',ret)
        return ret.flatten()
    '''updates valid rows'''
    def update_valid(self):
        inds = []
        for a in range(self.n_actions):
            base = a*self.samples_per_action 
            inds += range(base,base+self.mem_count[a])
        self.valid_inds = inds
        self.valid_mask[:] = 0
        self.valid_mask[inds] = 1

    '''returns inds and similarity of knn between a single state and an array of states'''
    def kernel(self,x,X):
        '''Gaussian'''
        dist = np.squeeze(cdist(np.expand_dims(x,0),X)/self.b)
        sim = norm.pdf(np.clip(dist,0,10))
        inds = np.argpartition(dist,self.k-1)[:self.k]
        assert np.all(sim>0), sim[sim<=0]
        return inds,sim

    '''for a given action, return next ind to be deleted'''
    def get_oldest_ind(self,a):
        return np.argmin(self.creation_time[a])

    ''' calc current value of single state s '''
    def get_value(self,s):
        cur_V = np.zeros((self.n_actions,))
        for a in range(self.n_actions):
            weights = np.zeros((self.samples_per_action,))
            inds,vals = self.kernel(s,self.S[a,:self.mem_count[a]])
            weights[inds] = vals[inds]
            cur_W_a = my_normalize(weights)
            cur_V[a] = cur_W_a.dot(self.R[a]+self.gamma*self.V[a])
        return cur_V.max(0)
    def add_tuple(self,t,s,a,r,sPrime,term):
        if self.mem_count[a] == self.samples_per_action:
            ind = self.get_oldest_ind(a)
        else:
            ind = self.mem_count[a]
        self.creation_time[a,ind] = t 
        self.S[a,ind] = s
        self.R[a,ind] = r
        self.NT[a,ind] = np.float32(not term)
        self.SPrime[a,ind] = sPrime
        if self.mem_count[a] < self.samples_per_action:
            self.mem_count[a] += 1
            self.update_valid()
            can_conflict = False
        else:
            can_conflict = True
        
        ''' don't calculate values until sufficient memories exist '''
        if self.warming:
            print('warming up!')
            ''' setup initial neighbor values'''
            if self.epsilon < 1 and np.all(self.mem_count >= self.k):
                print('done warming up!',self.mem_count)
                self.warming = False
                for v in self.valid_inds:
                    for act in range(self.n_actions):
                        knn_inds,sim = self.kernel(self.SPrime_view[v],self.S[act,:self.mem_count[act]])
                        self.W[act][v,knn_inds] = sim[knn_inds]
                        assert len(self.W[act].data[v]) == self.k, self.W[act].data[v]
                        worst_ind = np.argmin(sim[knn_inds])
                        assert worst_ind < self.k
                        self.WN_inds[act,v] = worst_ind
                        self.WN_vals[act,v] = sim[knn_inds[worst_ind]]
                        assert(sim[worst_ind] > 0)
            return

        row_ind = a*self.samples_per_action+ind
        assert(row_ind in self.valid_inds)
        for act in range(self.n_actions):
            #replace row
            knn_inds,sim = self.kernel(sPrime,self.S[act,:self.mem_count[act]])
            if can_conflict:
                self.W[act][row_ind,:] = 0
            self.W[act][row_ind,knn_inds] = sim[knn_inds]
            assert len(self.W[act].data[row_ind]) == self.k, self.W[act].data[row_ind]
            worst_ind = np.argmin(sim[knn_inds])
            assert worst_ind < self.k
            self.WN_inds[act,row_ind] = worst_ind
            self.WN_vals[act,row_ind] = sim[knn_inds[worst_ind]]
            assert np.all(self.WN_inds[act,self.valid_inds]>-1),str(a)+str(act)
        #----adjust columns
        #find rows relying on old memory in ind
        mask = self.valid_mask.copy()
        mask[row_ind] = 0 #don't overide a row we just added!
        if can_conflict:
            #TODO: replace valid_inds with all inds, since conflict requires full buffer
            conflict_inds = self.W[a][self.valid_inds,ind].nonzero()[0]
            if len(conflict_inds) > 0:
                vinds = np.asarray(self.valid_inds)
                mask[vinds[conflict_inds]] = 0 #don't overide a row we just added!
                for i in range(len(conflict_inds)):
                    v = self.valid_inds[conflict_inds[i]]
                    if v == row_ind:
                        continue
                    knn_inds,sim = self.kernel(self.SPrime_view[v],self.S[a,:self.mem_count[a]])
                    self.W[a][v,:] = 0 
                    self.W[a][v,knn_inds] = sim[knn_inds] 
                    assert len(self.W[a].data[v]) == self.k, self.W[a].data[v]
                    worst_ind = np.argmin(sim[knn_inds])
                    self.WN_inds[a,v] = worst_ind
                    assert worst_ind < self.k
                    self.WN_vals[a,v] = sim[knn_inds[worst_ind]]
        #find rows losing a neighbor
        _,sim = self.kernel(s,self.SPrime_view[self.valid_inds])
        mask[self.valid_inds] = np.logical_and(mask[self.valid_inds],self.WN_vals[a,self.valid_inds] < sim)
        assert np.all(self.WN_vals[a,mask]<sim[mask[self.valid_inds]])

        dead_inds = self.WN_inds[a,mask]
        array_of_lists = self.W[a].data[mask]
        list_of_ptrs = self.W[a].rows[mask]
        for i in range(len(dead_inds)):
            array_of_lists[i].pop(dead_inds[i])
            list_of_ptrs[i].pop(dead_inds[i])
        self.W[a][mask,ind] = np.expand_dims(sim[mask[self.valid_inds]],1) 
        for i in range(len(dead_inds)):
            assert len(array_of_lists[i]) == self.k, array_of_lists[i]
            new_worst_ind = np.argmin(array_of_lists[i])
            assert new_worst_ind < self.k, new_worst_ind
            self.WN_inds[a,dead_inds[i]] = new_worst_ind
            self.WN_vals[a,dead_inds[i]] = array_of_lists[i][new_worst_ind]

        #initial value
        #self.V[a,ind] = self.get_value(sPrime)
        self.V[a,ind] = 0
    def value_iteration(self):
        normed_W = []
        for a in range(self.n_actions):
            normed_W.append(my_normalize(self.W[a]))
        old_V = self.V_view.copy()
        temp_V = np.zeros((self.n_actions,self.n_samples))
        for i in range(self.update_ticks):
            for a in range(self.n_actions):
                temp_V[a] = bellman_op(normed_W[a],self.R[a],self.V[a],self.gamma,self.NT[a])
            self.V_view[:] = temp_V.max(0)
        change = np.abs(old_V-self.V_view).sum()
        print('change from update: ',change)
        return change
    def select_action(self,s):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions),0
        else:
            cur_V = np.zeros((self.n_actions,))
            for a in range(self.n_actions):
                weights = np.zeros((self.samples_per_action,))
                inds,vals = self.kernel(s,self.S[a,:self.mem_count[a]])
                weights[inds] = vals[inds]
                #TODO: replace with normalize and bellman_op?
                cur_W_a = (weights / weights.sum())
                cur_V[a] = cur_W_a.dot(self.R[a]+self.NT[a]*self.gamma*self.V[a])
            if np.all(cur_V==0):
                return np.random.randint(self.n_actions),0
            else:
                v_max_ind = np.squeeze(cur_V.argmax(0))
                return v_max_ind,cur_V[v_max_ind]

import gym
env = gym.make('CartPole-v0')
#env = gym.make('Pong-v0')
#env = simple_env.Simple()
agent = KADP(env)
s = agent.transform(env.reset())
cur_time = time.clock()
refresh = int(1e3)
tuples = []
cumr = 0
reward_per_episode = 0
Return = 0
anneal = int(2e3) 
anneal_schedule = np.linspace(1,.1,anneal)
for t in range(int(1e6)):
    '''
    anneal_state = t - 1000
    stop_anneal = int(1e5)
    agent.epsilon = min(1,max(.005,1-anneal_state/stop_anneal))
    '''
    agent.epsilon = anneal_schedule[min(t,anneal-1)]
    ''' select action'''
    if not agent.warming:
        a,val = agent.select_action(s)
    else:
        a = env.action_space.sample()
    ''' perform action, see result'''
    sPrime,r,term,_ = env.step(a)
    sPrime = agent.transform(sPrime)
    cumr += r
    Return +=r
    r = np.sign(r)
    ''' add to episodic memory '''
    tuples.append([t,s,a,r,sPrime,term])
    if term:
        print('new episode!')
        s = agent.transform(env.reset())
        reward_per_episode  = reward_per_episode*.99 + .01*Return
        Return = 0
    else:
        s = sPrime
    ''' update value function '''
    if t % agent.sample_ticks == 0:
        for i in range(len(tuples)):
            agent.add_tuple(*tuples[i])
        tuples = []
        if not agent.warming:
            agent.value_iteration()

    if t % refresh == 0:
        '''
        plt.figure(1)
        plt.clf()
        axes = plt.gca()
        axes.set_xlim([-4,4])
        axes.set_ylim([-4,4])
        #plt.scatter(agent.SPrime_view[:,0],agent.SPrime_view[:,1])
        plt.scatter(agent.SPrime_view[:,0],agent.SPrime_view[:,1],s=np.log(agent.V_view+1)*100,c=np.log(agent.V_view))
        plt.pause(.01)
        '''
        print(t,
                'time: ',time.clock()-cur_time,
                'reward: ',reward_per_episode,
                'memory state: ',agent.mem_count,
                'epsilon: ',agent.epsilon)
        cumr = 0
        cur_time = time.clock()



