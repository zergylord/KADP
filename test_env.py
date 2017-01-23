from Utils import simple_env
import numpy as np
env_choice = int(input('choose environment: Simple 1 or Grid 2'))
if env_choice == 1:
    env = simple_env.Simple(4)
else:
    env = simple_env.Grid(one_hot=True)
n_actions = env.action_space.n
A = list(range(n_actions))
s = env.reset()
while True:
    act = -1
    while act < 0 or act > 3:
        act = int(input('choose action'))-1
    act = env.oracle_policy()
    print(act)
    sPrime,r,_,_ = env.step(A[act])
    if env_choice == 1:
        print(env.encode(sPrime),r)
    else:
        print(sPrime.reshape(10,10))
