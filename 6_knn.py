import numpy as np
from scipy.stats import norm
from scipy import sparse
from scipy.spatial.distance import cdist,pdist
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import time
cur_time = time.clock()
n_samples = 2000
starting_points = 100
refresh = int(1e1)
sample_ticks = int(1e0)
update_ticks = int(1e0)
change_samples = True
test_n_samples = 100
n_actions = 4
rad_inc = 2*np.pi/n_actions
radius = .5
s_dim = 2 #assume this is fixed, since actions are rotations
b = .1
gamma = .9
epsilon = .1
display = True
interact = True
'''function defs'''
def get_transition(s,a):    
    sPrime =  s + np.asarray([radius*np.cos(rad_inc*a),radius*np.sin(rad_inc*a)])
    #sPrime += np.random.randn(2)*radius/2
    sPrime[sPrime > 4] = 4
    sPrime[sPrime < -4] = -4
    return sPrime
'''
only between singleton x and array X
'''
def kernel(x,X):
    '''Gaussian'''
    k = 5
    #sim = norm.pdf(np.squeeze(cdist(np.expand_dims(x,0),X))/b)
    dist = np.squeeze(cdist(np.expand_dims(x,0),X))
    '''
    vals,counts = np.unique(sim,return_counts=True)
    print(x,X)
    print(vals[0],counts[0])
    '''
    #inds = np.argpartition(sim,k-1)[-k:]
    inds = np.argpartition(dist,k-1)[:k]
    #sim = np.ones((X.shape[0],))
    sim = 1/(dist+1e-10)
    #inds = np.arange(sim.shape[0])
    assert(np.all(sim>0))
    return inds,sim

def normalize(W):
    W = W.copy()
    W_sum = W.sum(1)
    mask = W_sum != 0
    W[mask] = W[mask]/np.expand_dims(W_sum[mask],1)
    return W
def bellman_op(W,R,V):
    foo = normalize(W)
    bar = R+gamma*V
    '''
    for i in range(len(R)):
        if R[i] > 0:
            print(foo[:,i])
            '''
    return normalize(W).dot(R+gamma*V)
def get_reward(SPrime):
    if len(SPrime.shape) == 2:
        return np.float32((SPrime[:,0] > 1)
                *(SPrime[:,0] < 2)
                *(SPrime[:,1] > -1)
                *(SPrime[:,1] < 0))
    else:
        return np.float32((SPrime[0] > 1)
                *(SPrime[0] < 2)
                *(SPrime[1] > -1)
                *(SPrime[1] < 0))
if interact:
    plt.ion()
samples_per_action = int(n_samples/n_actions)
S = np.zeros((n_actions,samples_per_action,s_dim))
creation_time = np.asarray(np.tile(range(-samples_per_action,0),[n_actions,1]))
def get_oldest_ind(a):
    return np.argmin(creation_time[a])
SPrime = np.zeros((n_actions,samples_per_action,s_dim))
A = np.zeros((n_actions,samples_per_action),dtype=np.int32)
R = np.zeros((n_actions,samples_per_action))
V = np.zeros((n_actions,samples_per_action))
WN_inds = -1*np.ones((n_actions,n_samples)).astype('int')
WN_vals = np.zeros((n_actions,n_samples))
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
W = np.zeros((n_actions,n_samples,samples_per_action))

''' inds for view variables
for when memories arent yet full '''
valid_inds = []
valid_mask = np.zeros((n_samples,),dtype='bool')
def update_valid():
    global valid_inds,valid_mask
    inds = []
    for a in range(n_actions):
        base = a*samples_per_action 
        inds += range(base,base+mem_count[a])
    valid_inds = inds
    valid_mask[:] = 0
    valid_mask[inds] = 1

'''closure defs'''
def get_value(s):
    #calc current value of s
    cur_V = np.zeros((n_actions,))
    for a in range(n_actions):
        weights = np.zeros((samples_per_action,))
        inds,vals = kernel(s,S[a,:mem_count[a]])
        weights[inds] = vals[inds]
        cur_W_a = (weights / weights.sum())
        cur_V[a] = cur_W_a.dot(R[a]+gamma*V[a])
    return cur_V.max(0)
def add_tuple(t,s,a,r,sPrime):
    global WN_inds,WN_vals,creation_time,S,A,R,SPrime,W,V
    #cur_V_max = get_value(s)
    if mem_count[a] == samples_per_action:
        ind = get_oldest_ind(a)
    else:
        ind = mem_count[a]
    creation_time[a,ind] = t 
    S[a,ind] = s
    R[a,ind] = r
    SPrime[a,ind] = sPrime
    #print(np.nonzero(WN_inds[a,valid_inds]==-1)[0])
    mem_count[a] = min(mem_count[a] + 1,samples_per_action)
    update_valid()
    row_ind = a*samples_per_action+ind
    assert(row_ind in valid_inds)
    ''' sanity check'''
    '''
    W_sane = np.zeros((n_actions,n_samples,samples_per_action))
    for act in range(n_actions):
        for v in valid_inds:
            knn_inds,sim = kernel(SPrime_view[v],S[act,:mem_count[act]])
            W_sane[act,v,knn_inds] = sim[knn_inds]
    '''

    for act in range(n_actions):
        #replace row
        knn_inds,sim = kernel(sPrime,S[act,:mem_count[act]])
        W[act,row_ind,:] = 0
        W[act,row_ind,knn_inds] = sim[knn_inds]
        worst_ind = knn_inds[np.argmin(sim[knn_inds])]
        WN_inds[act,row_ind] = worst_ind
        WN_vals[act,row_ind] = sim[worst_ind]
        assert np.all(WN_inds[act,valid_inds]>-1),str(a)+str(act)
    #----adjust columns
    #find rows losing a neighbor
    _,sim = kernel(s,SPrime_view[valid_inds])
    mask = valid_mask.copy()
    assert np.count_nonzero(W[a,mask])/len(valid_inds) == 5
    mask[valid_inds] = WN_vals[a,valid_inds] < sim 
    mask[row_ind] = 0 #don't overide a row we just added!
    assert np.all(WN_vals[a,mask]<sim[mask[valid_inds]])
    W[a,mask,WN_inds[a,mask]] = 0 
    W[a,mask,ind] = sim[mask[valid_inds]] 
    #update worst neighbors
    foo = W[a,mask].copy()
    foo[foo==0] = np.nan
    WN_inds[a,mask] = np.nanargmin(foo,1) #lowest nonzero similarity
    WN_vals[a,mask] = W[a,mask,WN_inds[a,mask]]
    for act in range(n_actions):
        assert np.all(WN_inds[act,valid_inds]>-1),str(a)+str(act)

    #initial value
    V[a,ind] = 0
    #assert np.all(W_sane==W), str(np.nonzero(W_sane!=W))+str(W[W_sane!=W])+str(W_sane[W_sane!=W])+' action: '+str(a)+' row: '+str(row_ind)
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
        for j in range(test_n_samples):
            a,_ = select_action(test_S[j],0.1)
            test_S[j] = get_transition(test_S[j],a)
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
def select_action(cur_S,epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions),0
    else:
        cur_V = np.zeros((n_actions,))
        for a in range(n_actions):
            weights = np.zeros((samples_per_action,))
            inds,vals = kernel(cur_S,S[a,:mem_count[a]])
            weights[inds] = vals[inds]
            cur_W_a = (weights / weights.sum())
            cur_V[a] = cur_W_a.dot(R[a]+gamma*V[a])
        cur_V_max = cur_V.max(0)
        return np.squeeze(cur_V.argmax(0)),cur_V_max


'''initialize with points > k'''
for i in range(starting_points):
    a = np.random.randint(n_actions)
    s = (np.random.rand(s_dim)-.5)*2*3
    S[a,mem_count[a]] = s
    sPrime = get_transition(s,a)
    R[a,mem_count[a]] = get_reward(sPrime)
    if R[a,mem_count[a]] < 1:
        SPrime[a,mem_count[a]] = sPrime
    else:
        SPrime[a,mem_count[a]] = s
    mem_count[a] = min(mem_count[a] + 1,samples_per_action)
update_valid()
for a in range(n_actions):
    for v in valid_inds:
        knn_inds,sim = kernel(SPrime_view[v],S[a,:mem_count[a]])
        W[a,v,knn_inds] = sim[knn_inds]
        worst_ind = knn_inds[np.argmin(sim[knn_inds])]
        WN_inds[a,v] = worst_ind
        WN_vals[a,v] = sim[worst_ind]
        assert(sim[worst_ind] > 0)
'''main loop'''
cumr = 0
cur_s = (np.random.rand(s_dim)-.5)*2*3
greedy = True

'''
for j in range(100):
    V_view[:],change = value_iteration(W)
    print(change)
'''
for i in range(int(1e3)):
    if change_samples:
        tuples = []
        for j in range(sample_ticks):
            '''action selection'''
            #cur_s = (np.random.rand(s_dim)-.5)*2*3 #uniform sampling!
            #cur_a = np.random.randint(n_actions)
            cur_a,_ = select_action(cur_s,epsilon)
            cur_sPrime = get_transition(cur_s,cur_a)
            cur_r = get_reward(cur_sPrime)
            '''terminal via self transition and state reset'''
            if cur_r < 1:
                #add_tuple(i,cur_s,cur_a,cur_r,cur_sPrime)
                tuples.append([i,cur_s,cur_a,cur_r,cur_sPrime])
                cur_s = cur_sPrime
            else:
                print('yay!')
                cumr+=cur_r
                #add_tuple(i,cur_s,cur_a,cur_r,cur_s)
                tuples.append([i,cur_s,cur_a,cur_r,cur_s])
                cur_s = (np.random.rand(s_dim)-.5)*2*3
        for j in range(sample_ticks):
            for a in range(n_actions):
                assert(np.all(WN_inds[a,valid_inds]>-1))
            add_tuple(*tuples[j])
            for a in range(n_actions):
                assert np.all(WN_inds[a,valid_inds]>-1),str(cur_a)+str(a)
    for j in range(update_ticks):
        V_view[:],change = value_iteration(W)
    if display and i % refresh == 0:
        steps = i*sample_ticks
        if not greedy and change_samples and steps > 0*n_samples:
            greedy = True
            print('greedy time!')
            epsilon = .1
            sample_ticks = int(1e2)
            update_ticks = int(1e1)
        print(i,change,cumr,creation_time.min(1),time.clock()-cur_time)

        #cumr = 0
        cur_time = time.clock()
        assert(all(V_view==V.reshape(-1)))
        plt.clf()
        #'''
        axes = plt.gca()
        axes.set_xlim([-4,4])
        axes.set_ylim([-4,4])
        plt.scatter(SPrime_view[:,0],SPrime_view[:,1],s=100*((V)),c=V)
        #plt.scatter(SPrime_view[:,0],SPrime_view[:,1],c=V)
        #plt.hexbin(SPrime_view[:,0],SPrime_view[:,1],gridsize=15,extent=(-4,4,-4,4))
        #'''
        if interact:
            plt.pause(.01)
        else:
            plt.show()
        '''
        if change < 1e-5:
            print('done',i)
            break
        '''
plt.ioff()
plt.show()
plt.ion()
viz_trajectory()
