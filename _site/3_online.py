import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
interact = True
if interact:
    plt.ion()
n_samples = 10
n_actions = 4
s_dim = 2
b = .1
def kernel(x1,x2):
    return norm.pdf(np.clip(cdist(x1,x2)/b,0,10))
def get_weighting(x,x_i):
    weights = kernel(x,x_i)
    return weights / np.expand_dims(weights.sum(1),1)
def get_reward(S):
    if len(S.shape) == 1:
        return np.float32((S[0] > 1)
                *(S[0] < 2)
                *(S[1] > -1)
                *(S[1] < 0))
    else:
        return np.float32((S[:,0] > 1)
                *(S[:,0] < 2)
                *(S[:,1] > -1)
                *(S[:,1] < 0))
def get_transition(S):
    return S + np.random.randn(*(S.shape))

gamma = .9
S = np.random.randn(n_samples,s_dim)
R = get_reward(S)
SPrime = get_transition(S) #+ np.random.randn(n_samples,s_dim)
V = np.zeros((n_samples,))
W = get_weighting(SPrime,S)
refresh = int(1e1)
samples_per_step = 10
updates_per_step = 100
for i in range(int(1e3)):
    '''attach new sample(s)'''
    if samples_per_step > 0:
        S = np.append(S,np.random.randn(samples_per_step,s_dim),0)
        R = np.append(R,get_reward(S[-samples_per_step:,:]))
        if samples_per_step == 1:
            SPrime = np.append(SPrime,np.expand_dims(get_transition(S[-samples_per_step:,:]),0),0)
        else:
            SPrime = np.append(SPrime,get_transition(S[-samples_per_step:,:]),0)
        V = np.append(V,np.zeros((samples_per_step,)))
        W = get_weighting(SPrime,S)
    '''calculate new V using value iteration'''
    for j in range(updates_per_step):
        new_V = W.dot(R+gamma*V)
        change = np.abs(new_V-V).sum()
        V = new_V
        if change < 1e-4:
            print(i,j)
            break
    print(i,change)
    if i % refresh == 0:
        plt.clf()
        plt.scatter(SPrime[:,0],SPrime[:,1],s=100*(np.square(new_V)),c=new_V)
        '''
        test_S = (np.random.rand(1000,s_dim)-.5)*2*4
        test_W = get_weighting(test_S,S)
        test_V = test_W.dot(R+gamma*V)
        plt.scatter(test_S[:,0],test_S[:,1],s=100*(np.square(test_V)),c=test_V)
        '''
        if interact:
            plt.pause(.01)
        else:
            plt.show()

