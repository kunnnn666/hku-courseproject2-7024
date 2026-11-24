import torch
from torch import nn

class ResidualCNN(nn.Module):
    """Seven convolutions; signed residual output, no batch normalization."""
    def __init__(self):
        super().__init__()
        layers=[nn.Conv2d(3,32,3,padding=1),nn.ReLU()]
        for _ in range(5):layers.extend([nn.Conv2d(32,32,3,padding=1),nn.ReLU()])
        layers.append(nn.Conv2d(32,3,3,padding=1))
        self.layers=nn.Sequential(*layers)
    def forward(self,x):return self.layers(x)
