import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import cdist
n_samples = 100
n_actions = 4
s_dim = 1
def kernel(x1,x2):
    return norm.pdf(cdist(x1,x2))
def get_weighting(x,x_i):
    weights = kernel(x,x_i)
    return weights / np.expand_dims(weights.sum(1),1)
gamma = .9
S = np.random.randn(n_samples,s_dim)
R = np.random.randn(n_samples)
SPrime = np.random.randn(n_samples,s_dim)
V = np.zeros((n_samples,))
W = get_weighting(SPrime,S)
for i in range(int(1e3)):
    new_V = W.dot(R+gamma*V)
    change = np.abs(new_V-V).sum()
    V = new_V
    if change < 1e-4:
        print(i)
        break

