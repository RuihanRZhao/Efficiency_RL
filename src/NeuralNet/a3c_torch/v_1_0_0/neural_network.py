import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn.functional as F

# Define Actor-Critic Network
# class ActorCritic(nn.Module):
#     def __init__(self, input_size, output_size):  # 8 * n | 3 * n
#         super(ActorCritic, self).__init__()
#         self.actor = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=1),  # Adjust the number of channels and kernel size
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(64, output_size, kernel_size=1)
#         )
#         self.critic = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=1),  # Adjust the number of channels and kernel size
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 1, kernel_size=1)
#         )
#
#     def forward(self, x):
#         x = x.squeeze(0)
#         x = x.reshape(1, -1)
#         actor_output = self.actor(x)
#         critic_output = self.critic(x)
#         return actor_output, critic_output


class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size):  # 8 * n | 3 * n
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # Adjust the number of channels and kernel size
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.actor_action = nn.Sequential(
            nn.Linear(128 * 1 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * 12),
        )
        self.critic = nn.Sequential(
            nn.Linear(128 * 1 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output a single value for the value function
        )


    def forward(self, x):
        x = self.actor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.actor_action(x)
        value_G = self.critic(x)
        return x.reshape(3, 12), value_G