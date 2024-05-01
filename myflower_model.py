"""
@filename:myflower_model.py
@author:FengXing
@time:2024-05-01
"""
import torch.nn
from torch import nn


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 18, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 18,110,110
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(18, 36, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 36,53,53
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(36, 72, 4),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 72,25,25
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(72 * 25 * 25, 36 * 20 * 20, bias=True),
            nn.Sigmoid(),
            nn.Linear(36 * 20 * 20, 102 * 5 * 5, bias=True),
            nn.Sigmoid(),
            nn.Linear(102 * 5 * 5, 102, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear_stack(x)
        return x
