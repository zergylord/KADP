import numpy as np

class KADP(object):
    def __init__(self,env):
        self.starting_points = 1000
        self.sample_ticks = int(1e3)
        self.update_ticks = int(1e2)

        self.n_actions = env.action_space.n
        self.samples_per_action = 10000
        self.n_samples = self.n_actions*self.samples_per_action
        
        self.obs_dim = np.prod(env.observation_space.shape)
        self.s_dim = 64
        if self.obs_dim < self.s_dim:
            print('tiny state space, no projection.')
            self.s_dim = self.obs_dim
            self.transform = lambda x: x
        else:
            print('huge state space, random projection.')
            self.M = np.random.randn(self.obs_dim,self.s_dim) 
            self.transform = lambda x: np.dot(x,self.M)

    '''initialize with points > k per action'''
    s = env.reset()


import gym
env = gym.make('SpaceInvaders-v0')
agent = KADP(env)
s = env.reset().flatten()
foo = agent.transform(s)



