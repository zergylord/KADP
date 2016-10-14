import numpy as np
from scipy.stats import norm
from scipy import sparse
from scipy.spatial.distance import cdist,pdist
#from scipy.sparse import csc_matrix as sparse_matrix
from scipy.sparse import lil_matrix as sparse_matrix
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
#import numpy_indexed as npi
import time
cur_time = time.clock()
n_samples = 200000
starting_points = 100
total_steps = int(1e1)
refresh = int(1e0)
sample_ticks = int(1e3)
update_ticks = int(1e2)
softmax = False
change_samples = True
use_walls = False
use_transform = False
test_n_samples = 100
n_actions = 4
rad_inc = 2*np.pi/n_actions
radius = .25
s_dim = 2 #assume this is fixed, since actions are rotations
b = .1
gamma = .9
anneal = int(1e3)
epsilon = np.linspace(1,.1,anneal)
display = True
interact = True
debug = False
'''function defs'''
def get_transition(s,a):    
    sPrime =  s + np.asarray([radius*np.cos(rad_inc*a),radius*np.sin(rad_inc*a)])
    sPrime += np.random.randn(2)*radius
    if use_walls:
        #cross the vertical wall from 0,-1 upwards
        if sPrime[1] > -1 and sPrime[0] > 0 and s[0] <= 0:
            sPrime[0] = 0
        #cross the horizontal wall from -2,0 to 0,0 
        if sPrime[0] > -2 and sPrime[0] < 0 and sPrime[1] <= 0 and s[1] > 0:
            sPrime[1] = s[1]
    '''reset on wall hit, mainly to avoid duplicate states for testing purposes'''
    if debug:
        if np.any(sPrime > 4) or np.any(sPrime < -4):
            sPrime = (np.random.rand(s_dim)-.5)*2*3
    else:
        sPrime[sPrime > 4] = 4
        sPrime[sPrime < -4] = -4
    return sPrime
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
'''
only between singleton x and array X
'''
M = np.random.randn(2,64) 
def transform(x):
    #return np.power(x,3)
    return np.dot(x,M)
def kernel(x,X):
    if use_transform:
        x = transform(x)
        X = transform(X)
    k = 5
    '''Gaussian'''
    dist = np.squeeze(cdist(np.expand_dims(x,0),X)/b)
    if softmax:
        sim = norm.pdf(dist)
    else:
        sim = norm.pdf(np.clip(dist,0,10))
    inds = np.argpartition(dist,k-1)[:k]
    #sim = norm.pdf(dist)
    ''' inv dist'''
    '''
    dist = np.squeeze(cdist(np.expand_dims(x,0),X))
    sim = 1/(dist+1e-10)
    inds = np.argpartition(dist,k-1)[:k]
    '''
    '''cosine sim'''
    '''
    x = np.expand_dims(x,1)
    sim = np.squeeze(np.dot(X,x))
    denom = (np.linalg.norm(x)*np.linalg.norm(X,axis=1))
    sim /= denom
    sim = np.exp(sim)
    inds = np.argpartition(-sim,k-1)[:k]
    '''
    assert np.all(sim>0), sim[sim<=0]
    return inds,sim

def my_normalize(W):
    if len(W.shape) == 1:
        W = W.reshape(1,-1)
    return normalize(W,norm='l1',axis=1)
time_to_norm = 0
def bellman_op(W,R,V):
    global time_to_norm
    cur_time = time.clock()
    time_to_norm += (time.clock()-cur_time)
    res = W.dot(R+gamma*V)
    return res
if interact:
    plt.ion()
samples_per_action = int(n_samples/n_actions)
S = np.zeros((n_actions,samples_per_action,s_dim))
creation_time = np.asarray(np.tile(range(-samples_per_action,0),[n_actions,1]))
def get_oldest_ind(a):
    return np.argmin(creation_time[a])
SPrime = np.zeros((n_actions,samples_per_action,s_dim))
R = np.zeros((n_actions,samples_per_action))
V = np.zeros((n_actions,samples_per_action))
WN_inds = -1*np.ones((n_actions,n_samples)).astype('int')
WN_vals = np.zeros((n_actions,n_samples))
mem_count = np.zeros((n_actions,)).astype('int')
S_view = S.reshape(-1,s_dim)
R_view = R.reshape(-1)
SPrime_view = SPrime.reshape(-1,s_dim)
V_view = V.reshape(-1)
W = []
for act in range(n_actions):
    W.append(sparse_matrix((n_samples,samples_per_action),dtype='float64'))
    #W.append((np.zeros((n_samples,samples_per_action))))

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
        #cur_W_a = (weights / weights.sum())
        cur_W_a = my_normalize(weights)
        cur_V[a] = cur_W_a.dot(R[a]+gamma*V[a])
    return cur_V.max(0)
def get_state_pred_err(s,a,sPrime):
    inds,vals = kernel(s,S[a,:mem_count[a]])
    weights = np.zeros((samples_per_action,))
    weights[inds] = vals
    inds,vals = kernel(sPrime,SPrime[a,inds])
    return vals.min()

def get_value_grid():
    x = np.linspace(-4,4,100)
    y = np.linspace(4,-4,100)
    xv, yv = np.meshgrid(x,y)
    VG = np.zeros((100,100))
    for xi in range(100):
        for yi in range(100):
            VG[xi,yi] = get_value(np.asarray([xv[xi,yi],yv[xi,yi]]))
    return VG
time_to_add = 0
time_to_update_col = 0
def add_tuple(t,s,a,r,sPrime):
    global time_to_add,time_to_update_col,WN_inds,WN_vals,creation_time,S,R,SPrime,W,V
    cur_time = time.clock()
    #print(get_state_pred_err(s,a,sPrime))
    if mem_count[a] == samples_per_action:
        ind = get_oldest_ind(a)
    else:
        ind = mem_count[a]
    creation_time[a,ind] = t 
    S[a,ind] = s
    R[a,ind] = r
    SPrime[a,ind] = sPrime
    if mem_count[a] < samples_per_action:
        mem_count[a] += 1
        update_valid()
        can_conflict = False
    else:
        can_conflict = True
    row_ind = a*samples_per_action+ind
    assert(row_ind in valid_inds)
    if debug:
        ''' sanity check'''
        W_old = W.copy()
        W_sane = np.zeros((n_actions,n_samples,samples_per_action))
        for act in range(n_actions):
            for v in valid_inds:
                knn_inds,sim = kernel(SPrime_view[v],S[act,:mem_count[act]])
                W_sane[act,v,knn_inds] = sim[knn_inds]
    for act in range(n_actions):
        #replace row
        knn_inds,sim = kernel(sPrime,S[act,:mem_count[act]])
        if can_conflict:
            W[act][row_ind,:] = 0
        W[act][row_ind,knn_inds] = sim[knn_inds]
        assert len(W[act].data[row_ind]) == 5, W[act].data[row_ind]
        worst_ind = np.argmin(sim[knn_inds])
        assert worst_ind < 5
        WN_inds[act,row_ind] = worst_ind
        WN_vals[act,row_ind] = sim[knn_inds[worst_ind]]
        assert np.all(WN_inds[act,valid_inds]>-1),str(a)+str(act)
    #----adjust columns
    #find rows relying on old memory in ind
    col_time = time.clock()
    mask = valid_mask.copy()
    mask[row_ind] = 0 #don't overide a row we just added!
    if can_conflict:
        #TODO: replace valid_inds with all inds, since conflict requires full buffer
        conflict_inds = W[a][valid_inds,ind].nonzero()[0]
        if len(conflict_inds) > 0:
            vinds = np.asarray(valid_inds)
            mask[vinds[conflict_inds]] = 0 #don't overide a row we just added!
            for i in range(len(conflict_inds)):
                v = valid_inds[conflict_inds[i]]
                if v == row_ind:
                    continue
                knn_inds,sim = kernel(SPrime_view[v],S[a,:mem_count[a]])
                W[a][v,:] = 0 
                W[a][v,knn_inds] = sim[knn_inds] 
                assert len(W[a].data[v]) == 5, W[a].data[v]
                worst_ind = np.argmin(sim[knn_inds])
                WN_inds[a,v] = worst_ind
                assert worst_ind < 5
                WN_vals[a,v] = sim[knn_inds[worst_ind]]
    #find rows losing a neighbor
    _,sim = kernel(s,SPrime_view[valid_inds])
    mask[valid_inds] = np.logical_and(mask[valid_inds],WN_vals[a,valid_inds] < sim)
    assert np.all(WN_vals[a,mask]<sim[mask[valid_inds]])

    dead_inds = WN_inds[a,mask]
    array_of_lists = W[a].data[mask]
    list_of_ptrs = W[a].rows[mask]
    for i in range(len(dead_inds)):
        array_of_lists[i].pop(dead_inds[i])
        list_of_ptrs[i].pop(dead_inds[i])
    W[a][mask,ind] = np.expand_dims(sim[mask[valid_inds]],1) 
    for i in range(len(dead_inds)):
        assert len(array_of_lists[i]) == 5 , array_of_lists[i]
        new_worst_ind = np.argmin(array_of_lists[i])
        assert new_worst_ind < 5, new_worst_ind
        WN_inds[a,dead_inds[i]] = new_worst_ind
        WN_vals[a,dead_inds[i]] = array_of_lists[i][new_worst_ind]
    time_to_update_col += (time.clock()-col_time)
    '''
    emp_knn = np.count_nonzero(W[a,valid_inds])/len(valid_inds)
    assert  emp_knn == 5, emp_knn
    '''

    #update worst neighbors
    '''
    foo = W[a][mask].tocoo()
    _,inds = npi.group_by(foo.row).argmin(foo.data)
    WN_inds[a,mask] = inds #lowest nonzero similarity
    WN_vals[a,mask] = foo.data[inds]
    '''

    #initial value
    #V[a,ind] = get_value(sPrime)
    V[a,ind] = 0
    if debug:
        assert np.all(W_sane==W), str(np.nonzero(W_sane!=W))+str(W_old[W_sane!=W])+str(W[W_sane!=W])+str(W_sane[W_sane!=W])+' action: '+str(a)+' row: '+str(row_ind) + ' col: ' + str(ind)
    time_to_add += (time.clock()-cur_time)
def value_iteration(W):
    temp_V = np.zeros((n_actions,n_samples))
    for a in range(n_actions):
        temp_V[a] = bellman_op(W[a],R[a],V[a])
    new_V = temp_V.max(0)
    change = np.abs(new_V-V_view).sum()
    return new_V,change
def viz_trajectory():
    test_S = (np.random.rand(test_n_samples,s_dim)-.5)*2*3
    #test_S = np.random.randn(test_n_samples,s_dim)
    plt.figure(2)
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
s = (np.random.rand(s_dim)-.5)*2*4
for i in range(starting_points):
    #s = (np.random.rand(s_dim)-.5)*2*3
    a = np.random.randint(n_actions)
    S[a,mem_count[a]] = s
    sPrime = get_transition(s,a)
    R[a,mem_count[a]] = get_reward(sPrime)
    if R[a,mem_count[a]] < 1:
        SPrime[a,mem_count[a]] = sPrime
        s = sPrime
    else:
        SPrime[a,mem_count[a]] = s
        s = (np.random.rand(s_dim)-.5)*2*3
    mem_count[a] = min(mem_count[a] + 1,samples_per_action)
update_valid()
for a in range(n_actions):
    for v in valid_inds:
        knn_inds,sim = kernel(SPrime_view[v],S[a,:mem_count[a]])
        W[a][v,knn_inds] = sim[knn_inds]
        assert len(W[a].data[v]) == 5, W[a].data[v]
        worst_ind = np.argmin(sim[knn_inds])
        assert worst_ind < 5
        WN_inds[a,v] = worst_ind
        WN_vals[a,v] = sim[knn_inds[worst_ind]]
        assert(sim[worst_ind] > 0)

'''main loop'''
cumr = 0
cur_s = (np.random.rand(s_dim)-.5)*2*3
greedy = True

for i in range(total_steps):
    if change_samples:
        tuples = []
        for j in range(sample_ticks):
            '''action selection'''
            #cur_s = (np.random.rand(s_dim)-.5)*2*3 #uniform sampling!
            #cur_a = np.random.randint(n_actions)
            cur_a,_ = select_action(cur_s,epsilon[min(i,anneal-1)])
            cur_sPrime = get_transition(cur_s,cur_a)
            cur_r = get_reward(cur_sPrime)
            '''terminal via self transition and state reset'''
            if cur_r < 1:
                #add_tuple(i,cur_s,cur_a,cur_r,cur_sPrime)
                tuples.append([i,cur_s,cur_a,cur_r,cur_sPrime])
                cur_s = cur_sPrime
            else:
                cumr+=cur_r
                #add_tuple(i,cur_s,cur_a,cur_r,cur_s)
                tuples.append([i,cur_s,cur_a,cur_r,cur_s])
                cur_s = (np.random.rand(s_dim)-.5)*2*3
        for j in range(sample_ticks):
            add_tuple(*tuples[j])
    normed_W = []
    for act in range(n_actions):
        normed_W.append(my_normalize(W[act]))
    #V[:] = 0 #this surprisingly doesn't matter much!
    for j in range(update_ticks):
        V_view[:],change = value_iteration(normed_W)
    if display and i % refresh == 0:
        steps = i*sample_ticks
        if not greedy and change_samples and steps > 0*n_samples:
            greedy = True
            print('greedy time!')
            epsilon = .1
            sample_ticks = int(1e2)
            update_ticks = int(1e1)
        print(i,change,'tot r: ',cumr,'num mem: ',len(valid_inds),'tot time: ',time.clock()-cur_time, 'time to norm: ',time_to_norm, 'time to add: ',time_to_add,'time to col: ',time_to_update_col)
        time_to_norm = 0
        time_to_add = 0
        time_to_update_col = 0

        #cumr = 0
        cur_time = time.clock()
        assert all(V_view==V.reshape(-1)) , V_view[V_view!=V.reshape(-1)]
        plt.figure(0)
        VG = get_value_grid()
        plt.clf()
        plt.imshow(VG)
        plt.pause(.01)

        plt.figure(1)
        plt.clf()
        
        axes = plt.gca()
        axes.set_xlim([-4,4])
        axes.set_ylim([-4,4])
        plt.scatter(SPrime_view[:,0],SPrime_view[:,1],s=100*((V)),c=V)
        #plt.scatter(SPrime_view[:,0],SPrime_view[:,1],c=V)
        #plt.hexbin(SPrime_view[:,0],SPrime_view[:,1],gridsize=15,extent=(-4,4,-4,4))
        if interact:
            plt.pause(.01)
        else:
            plt.show()
        
        '''
        if change < 1e-5:
            print('done',i)
            break
        '''
'''
viz_trajectory()
plt.ioff()
plt.show()
'''
