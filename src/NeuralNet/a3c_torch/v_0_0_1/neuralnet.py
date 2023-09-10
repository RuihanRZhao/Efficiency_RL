import torch.nn as nn
import torch

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
        self.shit = nn.Sequential(
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
        actor_output = self.shit(x)
        value = self.critic(x)

        return actor_output.reshape(3, 12), value