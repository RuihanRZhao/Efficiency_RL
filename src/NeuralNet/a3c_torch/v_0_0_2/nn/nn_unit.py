import torch
import torch.nn as nn

class Data_Gather(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUpsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.conv(x)

class Action_Select(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
