import gym
from matplotlib import pyplot as plt
plt.ion()
env = gym.make('Pong-v0')
env.reset()

shist = []
for i in range(int(1e4)):
    s,r,term,_ = env.step(env.action_space.sample())
    shist.append(s)
    if term:
        print(r)
        s = env.reset()
    if r != 0:
        for j in range(5):
            plt.imshow(shist[-(j+1)])
            plt.pause(1)
        flag = True
