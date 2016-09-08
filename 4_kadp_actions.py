import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
n_samples = 10000
test_n_samples = 1000
n_actions = 3
rad_inc = 2*np.pi/n_actions
radius = .1
s_dim = 2 #assume this is fixed, since actions are rotations
b = .2
gamma = .9
epsilon = 0
'''function defs'''
def get_transition(S,A):    
    return S + np.asarray([radius*np.cos(rad_inc*A),radius*np.sin(rad_inc*A)]).transpose()
def kernel(x1,x2):
    return norm.pdf(np.clip(cdist(x1,x2)/b,0,10))
def get_weighting(x,x_i):
    weights = kernel(x,x_i)
    return weights / np.expand_dims(weights.sum(1),1)
def bellman_op(W,R,V):
    return W.dot(R+gamma*V)
interact = True
if interact:
    plt.ion()
S = (np.random.rand(n_samples,s_dim)-.5)*2*3
#S = np.random.randn(n_samples,s_dim)
A = np.random.randint(n_actions,size=(n_samples,))
SPrime = get_transition(S,A)
R = np.float32((SPrime[:,0] > 1)
        *(SPrime[:,0] < 2)
        *(SPrime[:,1] > -1)
        *(SPrime[:,1] < 0))
V = np.zeros((n_samples,))
test_V = np.zeros((test_n_samples,))
'''closure defs'''
def get_action_weightings():
    W = []
    for a in range(n_actions):
        W.append(get_weighting(SPrime,S[A==a]))
    return W
def value_iteration(W,A):
    temp_V = np.zeros((n_actions,len(A)))
    for a in range(n_actions):
        mask = A == a
        temp_V[a] = bellman_op(W[a],R[mask],V[mask])
    new_V = temp_V.max(0)
    change = np.abs(new_V-V).sum()
    return new_V,change
def viz_trajectory():
    test_S = (np.random.rand(test_n_samples,s_dim)-.5)*2*3
    #test_S = np.random.randn(test_n_samples,s_dim)
    for i in range(200):
        test_SPrime = np.zeros((n_actions,test_n_samples,s_dim))
        test_V = np.zeros((n_actions,test_n_samples))
        for a in range(n_actions):
            test_SPrime[a] = (get_transition(test_S,a))
            mask = A == a
            test_W_a = get_weighting(test_S,S[mask])
            test_V[a] = bellman_op(test_W_a,R[mask],V[mask])
        test_V_max = test_V.max(0)
        if np.random.rand() < epsilon:
            test_A = np.random.randint(n_actions,size=(test_n_samples,))
        else:
            test_A = test_V.argmax(0)
        if i % 2 == 0:
            plt.clf()
            axes = plt.gca()
            axes.set_xlim([-4,4])
            axes.set_ylim([-4,4])
            plt.scatter(test_S[:,0],test_S[:,1],s=10*((test_V_max)),c=test_V_max)
            plt.pause(.01)
        for s in range(test_n_samples):
            test_S[s] = test_SPrime[test_A[s],s]


'''main loop'''
W = get_action_weightings()
refresh = int(1e1)
for i in range(int(1e3)):
    V,change = value_iteration(W,A)
    print(i,change)
    if i % refresh == 0:
        plt.clf()
        #'''
        plt.scatter(SPrime[:,0],SPrime[:,1],s=100*((V)),c=V)
        #'''
        '''
        test_S = (np.random.rand(test_n_samples,s_dim)-.5)*2*3
        test_A = np.random.randint(n_actions,size=(test_n_samples,))
        test_SPrime = get_transition(test_S,test_A)
        for a in range(n_actions):
            test_mask = test_A == a
            mask = A == a
            test_W_a = get_weighting(test_S[test_mask],S[mask])
            print(test_V[test_mask].shape,test_W_a.shape,R[mask].shape)
            test_V[test_mask] = bellman_op(test_W_a,R[mask],V[mask])
        plt.scatter(test_SPrime[:,0],test_SPrime[:,1],s=10*((test_V)),c=test_V)
        '''
        if interact:
            plt.pause(.01)
        else:
            plt.show()
    if change < 1e-9:
        print(i)
        break
plt.ioff()
plt.show()
plt.ion()
viz_trajectory()
