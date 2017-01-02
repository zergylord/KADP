from Utils import simple_env
import numpy as np
env = simple_env.Grid(one_hot=True)
n_actions = env.action_space.n
A = list(range(n_actions))
s = env.reset()
while True:
    act = -1
    while act < 0 or act > 3:
        act = int(input('choose action'))-1
    print(act)
    sPrime,r,_,_ = env.step(A[act])
    print(sPrime.reshape(10,10))
