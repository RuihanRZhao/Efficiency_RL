'''
Reinforcement Learning (A3C)
Engine: Pytorch
Process: multi
Athor: Ryen Zhao
Date: 2023/Aug/03
'''

"""
import part
"""
# System Part
import math
import os
# Pytorch Part
import torch
import torch.nn as T_nn
import torch.nn.functional as T_nn_Function
import torch.multiprocessing as T_multiprocess
# TEMP GYM Part
import gym

"""
Varible Settings
"""

# Threads allowed
os.environ["OMP_NUM_THREADS"] = "8"

#
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000
MAX_EP_STEP = 200

# GYM Setting (TEMP)
env = gym.make('Pendulum-v0')
N_S = env.observation_space.shape[0]
# env Survey Space
N_A = env.action_space.shape[0]