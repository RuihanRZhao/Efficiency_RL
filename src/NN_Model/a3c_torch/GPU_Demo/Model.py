# Pytorch Part
import torch
import torch.nn as nn
import torch.nn.functional as function


class A3C_LSTM(nn.Module):
    def __init__(self, num_input, action_space, args):
        super(A3C_LSTM,self).__init__()
        self.hidden_size = args.hidden_size

        # data


        self.LSTM = nn.LSTMCell(1024, self.hidden_size)
        num_output = action_space.n
        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.actor_linear = nn.Linear(self.hidden_size, num_output)

