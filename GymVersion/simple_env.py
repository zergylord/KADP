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
class Simple(object):
    radius = .25
    @staticmethod
    def _new_state():
        return (np.random.rand(2)-.5)*2*4
    observation_space = ObservationSpace(2,lambda: Simple._new_state())
    @staticmethod
    def get_reward(SPrime):
        term = ((SPrime[0] > 1)
                 *(SPrime[0] < 2)
                 *(SPrime[1] > -1)
                 *(SPrime[1] < 0))
        return np.float32(term),term
        #return -1,term
    def __init__(self,n_actions = 4):
        self.reset()
        self.rad_inc = 2*np.pi/n_actions
        self.action_space = ActionSpace(n_actions)
    def reset(self):
        self.s = self._new_state()
        return self.s
    def get_transition(self,s,a):
        sPrime =  s + np.asarray([self.radius*np.cos(self.rad_inc*a),self.radius*np.sin(self.rad_inc*a)])
        sPrime += np.random.randn(2)*self.radius
        '''walls'''
        cross_hor_half = (s[1] < 0 and sPrime[1] > 0) or (s[1] > 0 and sPrime[1] < 0)
        cross_vert_half = (s[0] < 0 and sPrime[0] > 0) or (s[0] > 0 and sPrime[0] < 0)
        wall1 =  cross_hor_half and sPrime[0] > -3 and sPrime[0] < 3
        wall2 = cross_vert_half and sPrime[1] > -3 and sPrime[1] < 3
        if wall1 or wall2:
            sPrime = s
            r = 0
            term = True
        else:
            sPrime[sPrime > 4] = 4
            sPrime[sPrime < -4] = -4
            r,term = self.get_reward(sPrime)
        return sPrime,r,term
    def step(self,a):    
        sPrime,r,term = self.get_transition(self.s,a)
        self.s = sPrime
        return sPrime,r,term,False
