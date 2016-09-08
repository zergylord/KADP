import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
interact = True
if interact:
    plt.ion()
n_samples = 10000
n_actions = 4
s_dim = 2
b = 1
def kernel(x1,x2):
    return norm.pdf(np.clip(cdist(x1,x2)/b,0,10))
def get_weighting(x,x_i):
    weights = kernel(x,x_i)
    return weights / np.expand_dims(weights.sum(1),1)
gamma = .9
S = np.random.randn(n_samples,s_dim)
R = np.float32((S[:,0] > 1)
        *(S[:,0] < 2)
        *(S[:,1] > -1)
        *(S[:,1] < 0))
SPrime = S + np.random.randn(n_samples,s_dim)
V = np.zeros((n_samples,))
W = get_weighting(SPrime,S)
refresh = int(1e1)
for i in range(int(1e3)):
    new_V = W.dot(R+gamma*V)
    change = np.abs(new_V-V).sum()
    V = new_V
    print(i,change)
    if i % refresh == 0:
        plt.clf()
        #plt.scatter(SPrime[:,0],SPrime[:,1],s=100*(np.square(new_V)),c=new_V)
        test_S = (np.random.rand(1000,s_dim)-.5)*2*4
        test_W = get_weighting(test_S,S)
        test_V = test_W.dot(R+gamma*V)
        plt.scatter(test_S[:,0],test_S[:,1],s=100*(np.square(test_V)),c=test_V)
        if interact:
            plt.pause(.01)
        else:
            plt.show()
    if change < 1e-4:
        print(i)
        break

