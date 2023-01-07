import torch
import torch.nn as nn
import numpy as np

from hpf import *


# Xu_net

class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()

        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)

        return output

class AbsWrapper(nn.Module):
    def forward(self, x):
        x = torch.abs(x)
        return x


class XuNet(nn.Module):
    def __init__(self):
        super(XuNet, self).__init__()

        self.hpf = HPF_srm6()


        self.group1 = nn.Sequential(
            nn.Conv2d(6, 8, kernel_size=5, stride=1, padding=2, bias = False),
            AbsWrapper(),
            nn.BatchNorm2d(8, momentum=0.1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2,)
        )

        self.group3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )

        self.group4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )

        self.group5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=256 // 16, stride=256 // 16)
        )

        self.fc1 = nn.Linear(128, 2)

    def forward(self, input):
        output = input

        output = self.hpf(output)

        output = self.group1(output)
        output = self.group2(output)
        output = self.group3(output)
        output = self.group4(output)
        output = self.group5(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)

        return output

