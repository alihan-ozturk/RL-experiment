import numpy as np
import gym
import time
from IPython.display import clear_output
from gym.envs.toy_text.frozen_lake import generate_random_map
from utils import argmax

Q = np.load("Q_table.npy")


random_map = generate_random_map(size=20, p=0.4)
env = gym.make("FrozenLake-v1", desc=random_map)
state = env.reset()

episode_length=0
while True:
    action = np.argmax(Q[state])
    next_state, reward, done, info = env.step(action)
    print(episode_length)
    print(env.render(mode="ansi"))
    time.sleep(.5)
    if done:
        print(reward)
        break
    time.sleep(0.2)
    episode_length += 1
