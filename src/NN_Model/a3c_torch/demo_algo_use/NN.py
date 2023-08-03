'''
Reinforcement Learning (A3C)
Engine: Pytorch
Process: multi

File Name: NN
Describe:
    Class of Neural Network of A3C

Athor: Ryen Zhao
Date: 2023/Aug/03
'''

"""
import part
"""
# System Part
import math

# Pytorch Part
import torch
import torch.nn as T_nn
import torch.nn.functional as T_nn_Function

# Functions
import utils
class Neural_Network(T_nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Neural_Network, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.a1 = T_nn.Linear(s_dim, 200)
        self.Multi = T_nn.Linear(200, a_dim)
        self.sigma = T_nn.Linear(200, a_dim)
        self.c1 = T_nn.Linear(s_dim, 100)
        self.v = T_nn.Linear(100, 1)
        utils.set_init([self.a1, self.Multi, self.sigma, self.c1, self.v])

        # initialize parameter
        self.distribution = torch.distributions.Normal

    def forward(self, arg_x):
        a1 = T_nn_Function.relu6(self.a1(arg_x))
        Multi = 2 * T_nn_Function.tanh(self.Multi(a1))
        sigma = T_nn_Function.softplus(self.sigma(a1))+0.001    # avoid: = 0
        c1 = T_nn_Function.relu6(self.c1(arg_x))
        values = self.v(c1)
        return Multi, sigma, values

    def choose_action(self, s):
        self.training = False
        mu, sigma, _ = self.forward(s)
        action = self.distribution(mu.view(1, -1).data, sigma.view(1, -1).data)
        # distribution choose action, match the "Action Possibility" in loss_function
        return action.sample().numpy()

    def loss_function(self, s, a, v_t):
        self.train()
        mu, sigma, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)  # Error function of critic

        m = self.distribution(mu, sigma)

        # Generate normal distribution
        log_prob = m.log_prob(a)

        '''
        Action Possibility
        Expected Value of Action: log_prob * td
        '''
        # Add error to increase exploratory, similar exploration in greedy algorithm
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(m.scale)
        exp_v = log_prob * td.detach() + 0.005 * entropy
        a_loss = -exp_v
        total_loss = (a_loss + c_loss).mean()

        return total_loss



