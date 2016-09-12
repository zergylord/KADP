import gym
env = gym.make('DoomBasic-v0')
env.reset()
for i in range(1000):
    env.step(env.action_space.sample())
    env.render()
