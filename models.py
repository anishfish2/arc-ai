import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy


class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(900 + 4, 512)
        self.fc2 = torch.nn.Linear(512, 128)
        self.fc3 = torch.nn.Linear(128, 12)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, **kwargs):
        x = torch.nn.functional.leaky_relu(self.conv1(x))
        x = torch.nn.functional.leaky_relu(self.conv2(x))
        x = torch.nn.functional.leaky_relu(self.conv3(x))
        x = torch.nn.functional.leaky_relu(self.conv4(x))
        x = torch.nn.functional.leaky_relu(self.conv5(x))
        x = self.fc1(torch.cat((
                x.view(1, 900),  # Assuming x.view(1, 900) is correct shape
                torch.tensor([[kwargs['height_index']]]).to(self.device),  # Convert to 2D tensor
                torch.tensor([[kwargs['width_index']]]).to(self.device),   # Convert to 2D tensor
                torch.tensor([[kwargs['height']]]).to(self.device),        # Convert to 2D tensor
                torch.tensor([[kwargs['width']]]).to(self.device)          # Convert to 2D tensor
            ), 1))
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.softmax(x)
        return x
    