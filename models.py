import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy


class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        # self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.fc1 = torch.nn.Linear(32 * 7 * 7, 128)
        # self.fc2 = torch.nn.Linear(128, 19)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        # x = self.pool(x)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        # x = self.pool(x)
        # x = x.view(-1, 32 * 7 * 7)
        # x = torch.nn.functional.relu(self.fc1(x))
        # x = self.fc2(x)
        return x
    