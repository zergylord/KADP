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
num_cells = 8
def state2vec(s):
    one_hot = np.zeros(num_cells*2)
    one_hot[int(s[0])] = 1
    one_hot[num_cells-1+int(s[0])] = 1
    return one_hot
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
    n = 3
    '''
    def encode(obs):
        return odd_root(obs*Simple.limit**n,n)
    def decode(s):
        return (s**n)/(Simple.limit**n)
    '''
    @staticmethod
    def encode(obs):
        return obs*Simple.limit
    @staticmethod
    def decode(s):
        return s/Simple.limit
    radius = .25
    limit = 4
    @staticmethod
    def _new_state():
        return Simple.decode((np.random.rand(2)-.5)*2*Simple.limit)
    observation_space = ObservationSpace(2,lambda: Simple._new_state())
    def get_reward(self,SPrime):
        term = ((SPrime[0] >= self.x_goal)
                 *(SPrime[0] < self.x_goal+self.radius*4)
                 *(SPrime[1] >= self.y_goal)
                 *(SPrime[1] < self.y_goal+self.radius*4))
        return np.float32(term),term
        #return -1,term
    def gen_goal(self):
        self.x_goal = np.random.rand()*(self.limit*2-1)-self.limit
        self.y_goal = np.random.rand()*(self.limit*2-1)-self.limit
    def __init__(self,n_actions = 4):
        self.reset()
        self.gen_goal()
        self.rad_inc = 2*np.pi/n_actions
        self.action_space = ActionSpace(n_actions)
    def reset(self):
        self.s = self._new_state()
        return self.s
    def get_transition(self,obs,a):
        s = Simple.encode(obs)
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
                r = 0
            else:
                r,term = self.get_reward(sPrime)
            sPrime = Simple.decode(sPrime)
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
class Cycle(object):
    step_size = 1.0
    goal_size = 1.0
    cycle_size = 40
    def encode(self,obs):
        if self.one_hot:
            return np.argmax(obs,-1)
        else:
            return obs*Cycle.cycle_size
    def decode(self,s):
        if self.one_hot:
            zeros = np.zeros((Cycle.cycle_size,))
            zeros[int(s)] = 1
            return zeros
        else:
            return s/Cycle.cycle_size
    def gen_goal(self):
        self.goal = np.random.randint(self.cycle_size-1)
        print(self.goal)
    def _new_state(self):
        #return self.decode(np.random.rand()*Cycle.cycle_size)
        return self.decode(np.random.randint(Cycle.cycle_size))
    def get_reward(self,SPrime):
        in_goal = SPrime >= self.goal and SPrime < (self.goal+Cycle.goal_size)
        antigoal = self.goal+1
        in_antigoal = SPrime >= antigoal and SPrime < (antigoal+Cycle.goal_size)
        if in_goal:
            r = 1
        elif in_antigoal:
            r = -.5
        else:
            r = -.1#*(1-np.abs(self.goal-SPrime)/self.cycle_size)
        return r,False
    def __init__(self,n_actions = 2,one_hot = True):
        self.one_hot = one_hot
        if self.one_hot:
            s_dim = Cycle.cycle_size
        else:
            s_dim = 1
        self.gen_goal()
        self.observation_space = ObservationSpace(s_dim,lambda: self._new_state())
        self.reset()
        self.action_space = ActionSpace(n_actions)
    def reset(self):
        self.s = self._new_state()
        return self.s
    def get_transition(self,obs,a):
        s = self.encode(obs)
        if a == 0:
            sPrime = (s + Cycle.step_size) % Cycle.cycle_size
        else:
            sPrime = (s - Cycle.step_size) % Cycle.cycle_size
        r,term = self.get_reward(sPrime)
        sPrime = self.decode(sPrime)
        return sPrime,r,term
    def step(self,a):    
        sPrime,r,term = self.get_transition(self.s,a)
        self.s = sPrime
        return sPrime,r,term,False

class Grid(object):
    side_size = 10
    #for completeness with the continous version
    radius = .25
    rad_inc = 2*np.pi/4
    def encode(self,obs):
        if self.one_hot:
            ind = np.flatnonzero(obs)[0]
            return np.asarray([ind//self.side_size,ind%self.side_size])
        else:
            return obs*Grid.side_size
    def decode(self,s):
        if self.one_hot:
            zeros = np.zeros((Grid.side_size**2,))
            zeros[s[0]*self.side_size+s[1]] = 1
            return zeros
        else:
            return s/Grid.side_size
    def gen_goal(self):
        self.goal = np.random.randint(self.side_size,size=(2,))
        print(self.goal)
    def _new_state(self):
        return self.decode(np.random.randint(Grid.side_size,size=(2,)))
    def get_reward(self,SPrime):
        in_goal = np.all(SPrime == self.goal)
        if in_goal:
            r = 1
        else:
            r = -.1
        return r,False
    def __init__(self,one_hot = True):
        self.one_hot = one_hot
        if self.one_hot:
            s_dim = Grid.side_size**2
        else:
            s_dim = 2
        n_actions = 4
        self.gen_goal()
        self.observation_space = ObservationSpace(s_dim,lambda: self._new_state())
        self.reset()
        self.action_space = ActionSpace(n_actions)
    def reset(self):
        self.s = self._new_state()
        return self.s
    def get_transition(self,obs,a):
        s = self.encode(obs)
        max_s = np.asarray([self.side_size,self.side_size])
        if a == 0:#right
            sPrime = (s + [1,0]) %max_s
        elif a == 1:#up
            sPrime = (s - [0,1]) %max_s
        elif a == 2:#left
            sPrime = (s - [-1,0]) %max_s
        elif a == 3:#down
            sPrime = (s + [0,-1]) %max_s
        r,term = self.get_reward(sPrime)
        sPrime = self.decode(sPrime)
        return sPrime,r,term
    def step(self,a):    
        sPrime,r,term = self.get_transition(self.s,a)
        self.s = sPrime
        return sPrime,r,term,False
