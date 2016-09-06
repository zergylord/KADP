import numpy as np
n_samples = 100
n_actions = 4
s_dim = 1
def kernel(x1,x2):
    return max(0,10-np.abs(x1-x2))
def weighting(x,x_i):
    weights = np.zeros((len(x_i),))
    for i in len(x_i):
        weights[i] = kernel(x,x_i[i])
    return weights / weights.sum() 
S = np.random.randn(n_samples,s_dim)
R = np.random.randn(n_samples)
SPrime = np.random.randn(n_samples,s_dim)
V = np.zeros((n_samples,))

V = V
