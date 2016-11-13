import numpy as np

class ActionSpace(object):
    def __init__(self,n):
        self.n = n
        self.sample = lambda: np.random.randint(n)
class ObservationSpace(object):
    def __init__(self,shape,sample = None):
        self.shape = shape
        if sample is None:
            self.sample = lambda: np.random.randn(shape)
        else:
            self.sample = sample
def odd_root(x,n=3):
    return np.power(abs(x),float(1)/n)*np.sign(x)
n = 3
'''
def encode(obs):
    return odd_root(obs*Simple.limit**n,n)
def decode(s):
    return (s**n)/(Simple.limit**n)
'''
def encode(obs):
    return obs*Simple.limit
def decode(s):
    return s/Simple.limit
'''
M = np.random.randn(2,2)
b = np.random.randn(2)*0
W = np.linalg.inv(M)
d = -b
def encode(obs):
    return np.dot(obs,M)+b
def decode(s):
    return np.dot(s,W)+d
'''
class Simple(object):
    radius = .5 #.25
    limit = 4
    @staticmethod
    def _new_state():
        return decode((np.random.rand(2)-.5)*2*Simple.limit)
    observation_space = ObservationSpace(2,lambda: Simple._new_state())
    @staticmethod
    def get_reward(SPrime):
        term = ((SPrime[0] > 2)
                 *(SPrime[0] < 3)
                 *(SPrime[1] > 2)
                 *(SPrime[1] < 3))
        return np.float32(term),term
        #return -1,term
    def __init__(self,n_actions = 4):
        self.reset()
        self.rad_inc = 2*np.pi/n_actions
        self.action_space = ActionSpace(n_actions)
    def reset(self):
        self.s = self._new_state()
        return self.s
    def get_transition(self,obs,a):
        s = encode(obs)
        assert np.all(s>=-4),s[s<-4]
        assert np.all(s<=4),s[s>4]
        sPrime =  s + np.asarray([self.radius*np.cos(self.rad_inc*a),self.radius*np.sin(self.rad_inc*a)])
        #sPrime += np.random.randn(2)*self.radius*.1
        '''walls'''
        cross_hor_half = (s[1] < 0 and sPrime[1] > 0) or (s[1] > 0 and sPrime[1] < 0)
        cross_vert_half = (s[0] < 0 and sPrime[0] > 0) or (s[0] > 0 and sPrime[0] < 0)
        wall1 =  cross_hor_half and sPrime[0] > -3 and sPrime[0] < 3
        wall2 = cross_vert_half and sPrime[1] > -3 and sPrime[1] < 3
        if False and wall1: #or wall2):
            sPrime = obs
            r = 0
            term = False
        else:
            if np.any(sPrime > self.limit) or np.any(sPrime < -self.limit):
                sPrime[sPrime > self.limit] = self.limit
                sPrime[sPrime < -self.limit] = -self.limit
                term = True
                r = -1
            else:
                r,term = self.get_reward(sPrime)
            sPrime = decode(sPrime)
        return sPrime,r,term
    def step(self,a):    
        sPrime,r,term = self.get_transition(self.s,a)
        self.s = sPrime
        return sPrime,r,term,False
'''
from matplotlib import pyplot as plt
env = Simple()
x = np.linspace(-env.limit,env.limit,30)
y = np.linspace(env.limit,-env.limit,30)
xv, yv = np.meshgrid(x,y)
count = 0
mb_s = np.zeros((900,2))
for xi in range(30):
    for yi in range(30):
        mb_s[count,:] = np.asarray([xv[xi,yi],yv[xi,yi]])
        count +=1
mb_s = decode(mb_s)
mb_s = encode(mb_s)


plt.scatter(mb_s[:,0],mb_s[:,1])
plt.show()
'''
