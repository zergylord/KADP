import numpy as np
from scipy.stats import norm
from scipy import sparse
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import time
cur_time = time.clock()
n_samples = 1000
test_n_samples = 1000
n_actions = 4
rad_inc = 2*np.pi/n_actions
radius = .1
s_dim = 2 #assume this is fixed, since actions are rotations
b = .1
gamma = .9
epsilon = 0
change_samples = True
display = True
interact = True
'''function defs'''
def get_transition(S,A):    
    return S + np.asarray([radius*np.cos(rad_inc*A),radius*np.sin(rad_inc*A)]).transpose()
'''
only between singleton x and array X
'''
def kernel(x,X):
    '''Gaussian'''
    k = 4
    sim = norm.pdf(np.clip(cdist(x,X)/b,0,10))
    inds = np.argpartition(sim,k-1)[:k]
    return inds,sim


def bellman_op(W,R,V):
    return (W/np.expand_dims(W.sum(1),1)).dot(R+gamma*V)
def get_reward(SPrime):
    return np.float32((SPrime[:,0] > 1)
            *(SPrime[:,0] < 2)
            *(SPrime[:,1] > -1)
            *(SPrime[:,1] < 0))
if interact:
    plt.ion()
samples_per_action = int(n_samples/n_actions)
S = (np.random.rand(n_actions,samples_per_action,s_dim)-.5)*2*3
creation_time = np.asarray(np.tile(range(-samples_per_action,0),[n_actions,1]))
def get_oldest_ind(a):
    return np.argmin(creation_time[a])
SPrime = np.zeros((n_actions,samples_per_action,s_dim))
A = np.zeros((n_actions,samples_per_action),dtype=np.int32)
R = np.zeros((n_actions,samples_per_action))
V = np.zeros((n_actions,samples_per_action))
NW_inds = np.zeros((n_actions,samples_per_action)).astype('int')
NW_vals = np.zeros((n_actions,samples_per_action))
mem_count = np.zeros((n_actions,)).astype('int')
for a in range(n_actions):
    SPrime[a] = get_transition(S[a],a)
    R[a] = get_reward(SPrime[a])
    SPrime[a,R[a]==1,:] = S[a,R[a]==1,:]
    A[a] = a
S_view = S.reshape(-1,s_dim)
A_view = A.reshape(-1)
R_view = R.reshape(-1)
SPrime_view = SPrime.reshape(-1,s_dim)
V_view = V.reshape(-1)
''' inds for view variables
for when memories arent yet full '''
def update_memory_vars(X):
    global valid_inds,mem_count
    inds = []
    for a in n_actions:
        base = a*samples_per_action 
        inds += range(base,base+mem_count[a])
    return inds

'''closure defs'''
def get_initial_weightings():
    W = np.zeros((n_actions,n_samples,samples_per_action))
    for a in range(n_actions):
        W[a][ = (kernel(SPrime_view[valid_inds],S[a]))
    return W
def add_tuple(t,s,a,r,sPrime):
    global creation_time,S,A,R,SPrime,W,V
    ind = get_oldest_ind(a)
    creation_time[a,ind] = t 
    S[a,ind] = s
    R[a,ind] = r
    SPrime[a,ind] = sPrime
    for act in range(n_actions):
        #replace row
        knn_inds,sim = kernel(sPrime,S[act])
        #cache for updating columns
        if act == a:
            a_sim = sim
        W[act,a*samples_per_action+ind,:] = 0
        W[act,a*samples_per_action+ind,knn_inds] = sim[knn_inds]
    #----adjust columns
    #find rows losing a neighbor
    mask = WN_vals[a]< a_sim
    W[a,mask,WN_inds[a,mask]] = 0 
    W[a,mask,ind] = a_sim[mask] 
    #update worst neighbors
    NW_inds[a,mask] = W[a,mask].argmax(1)
    NW_vals[a,mask] = W[a,mask,NW_inds[a,mask]]

    V[a,ind] = 0 
    #V[oldest_S_ind[a]] = bellman_op(new_row,r,V[A==a])
    mem_count[a] = min(mem_count[a] + 1,samples_per_action)
def value_iteration(W):
    temp_V = np.zeros(W.shape[:-1])
    for a in range(n_actions):
        temp_V[a] = bellman_op(W[a],R[a],V[a])
    new_V = temp_V.max(0)
    change = np.abs(new_V-V_view).sum()
    return new_V,change
def viz_trajectory():
    test_S = (np.random.rand(test_n_samples,s_dim)-.5)*2*3
    #test_S = np.random.randn(test_n_samples,s_dim)
    for i in range(200):
        test_A,test_V = select_action(test_S)
        test_SPrime = get_transition(test_S,test_A)
        if i % 2 == 0:
            plt.clf()
            '''
            axes = plt.gca()
            axes.set_xlim([-4,4])
            axes.set_ylim([-4,4])
            plt.scatter(test_S[:,0],test_S[:,1],s=10*((test_V_max)),c=test_V_max)
            '''
            plt.hexbin(test_S[:,0],test_S[:,1],gridsize=15,extent=(-4,4,-4,4))
            plt.pause(.01)
        test_S = test_SPrime
def select_action(cur_S):
    cur_V = np.zeros((n_actions,len(cur_S)))
    for a in range(n_actions):
        weights = np.zeros((samples_per_action,))
        inds,vals = kernel(cur_S,S[a])
        weights[inds] = vals
        cur_W_a = (weights / np.expand_dims(weights.sum(1),1))
        cur_V[a] = bellman_op(cur_W_a,R[a],V[a])
    cur_V_max = cur_V.max(0)
    return np.squeeze(cur_V.argmax(0)),cur_V_max


'''main loop'''
W = get_action_weightings()
refresh = int(1e0)
cur_s = (np.random.rand(1,s_dim)-.5)*2*3
for i in range(int(1e1)):
    if change_samples:
        '''action selection'''
        cur_a,_ = select_action(cur_s)
        cur_sPrime = get_transition(cur_s,cur_a)
        cur_r = get_reward(cur_sPrime)
        '''terminal via self transition and state reset'''
        if cur_r < 1:
            add_tuple(i,cur_s,cur_a,cur_r,cur_sPrime)
            cur_s = cur_sPrime
        else:
            print('yay!')
            add_tuple(i,cur_s,cur_a,cur_r,cur_s)
            cur_s = (np.random.rand(1,s_dim)-.5)*2*3
    for j in range(10):
        V_view[:],change = value_iteration(W)
    if display and i % refresh == 0:
        print(i,change,creation_time.min(1))
        assert(all(V_view==V.reshape(-1)))
        plt.clf()
        #'''
        plt.scatter(SPrime_view[:,0],SPrime_view[:,1],s=100*((V)),c=V)
        #plt.scatter(SPrime[:,0],SPrime[:,1],c=V)
        #plt.hexbin(SPrime_view[:,0],SPrime_view[:,1],gridsize=15,extent=(-4,4,-4,4))
        #'''
        if interact:
            plt.pause(.01)
        else:
            plt.show()
        if change < 1e-5:
            print('done',i)
            break
print(time.clock()-cur_time)
'''
plt.ioff()
plt.show()
plt.ion()
viz_trajectory()
'''
