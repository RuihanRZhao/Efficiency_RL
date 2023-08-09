import os
import gymnasium as gym

# Threads allowed
os.environ["OMP_NUM_THREADS"] = "1"

#
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 100000
MAX_EP_STEP = 200

# GYM Setting (TEMP)
env = gym.make('Pendulum-v1')
N_S = env.observation_space.shape[0]
# env Survey Space
N_A = env.action_space.shape[0]