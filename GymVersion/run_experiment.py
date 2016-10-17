import time
import gym_ple
import gym
import numpy as np
from KADP import KADP
import sys
import simple_env
if len(sys.argv) < 2:
    print('defaulting to simple environment!')
    env = simple_env.Simple()
    agent = KADP(env,n_buffer_obs = 1)
    anneal = int(1e3) 
    anneal_schedule = np.linspace(1,.005,anneal)
elif sys.argv[1] == 'Catch':
    env = gym.make('Catcher-v0')
    agent = KADP(env,b=1e0)
    anneal = int(1e5) 
    anneal_schedule = np.linspace(1,.005,anneal)
elif sys.argv[1] == 'Pong':
    env = gym.make('Pong-v0')
    agent = KADP(env,b=1e0)
    anneal = int(1e5) 
    anneal_schedule = np.linspace(1,.005,anneal)
#env = gym.make('CartPole-v0') #2e3 to 0 epsilon, 15 knn
#env = gym.make('Catcher-v0')
s = agent.transform(env.reset())
cur_time = time.clock()
total_steps = int(1e6)
refresh = int(1e3)
tuples = []
cumr = 0
episode_count = 0
reward_per_episode = -1
last_return = 0
Return = 0
for t in range(total_steps):
    '''
    anneal_state = t - 1000
    stop_anneal = int(1e5)
    agent.epsilon = min(1,max(.005,1-anneal_state/stop_anneal))
    '''
    agent.epsilon = anneal_schedule[min(t,anneal-1)]
    ''' select action'''
    if not agent.warming:
        a,val = agent.select_action(s)
    else:
        a = env.action_space.sample()
    ''' perform action, see result'''
    sPrime,r,term,_ = env.step(a)
    sPrime = agent.transform(sPrime)
    cumr += r
    Return +=r
    #r = np.sign(r)
    '''
    if r != 0:
        term = True
    '''
    ''' add to episodic memory '''
    tuples.append([t,s,a,r,sPrime,term])
    if term:
        #print('new episode!')
        s = agent.transform(env.reset())
        reward_per_episode  = reward_per_episode*.95 + .05*Return
        episode_count += 1
        last_return = Return
        Return = 0
    else:
        s = sPrime
    ''' update value function '''
    if t % agent.sample_ticks == 0:
        for i in range(len(tuples)):
            agent.add_tuple(*tuples[i])
        tuples = []
        if not agent.warming:
            agent.value_iteration()

    if t % refresh == 0:
        '''
        plt.figure(1)
        plt.clf()
        axes = plt.gca()
        axes.set_xlim([-4,4])
        axes.set_ylim([-4,4])
        #plt.scatter(agent.SPrime_view[:,0],agent.SPrime_view[:,1])
        plt.scatter(agent.SPrime_view[:,0],agent.SPrime_view[:,1],s=np.log(agent.V_view+1)*100,c=np.log(agent.V_view))
        plt.pause(.01)
        '''
        print(t,
                'time: ',time.clock()-cur_time,
                'reward: ',reward_per_episode,
                'episode: ',episode_count,
                'memory state: ',agent.mem_count,
                'epsilon: ',agent.epsilon,
                'avg sim: ',agent.avg_sim)
        cumr = 0
        cur_time = time.clock()
