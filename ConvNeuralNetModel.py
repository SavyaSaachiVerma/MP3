import torch
import torch.nn as nn


class ConvNeuralNet(nn.Module):
    """ The deep convolution neural network model.
        All the models have to inherit the nn.Module."""

    def __init__(self):
        super(ConvNeuralNet, self).__init__()

        self.layers = nn.Sequential(
            # Convolution Layer - 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=1,
                      padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Convolution Layer - 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                      padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.05),
            # Convolution Layer - 3
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                      padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Convolution Layer - 4
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                      padding=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.05),
            # Convolution Layer - 5
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                      padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Convolution Layer - 6
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      padding=0),
            nn.Dropout(p=0.05),
            # Convolution Layer - 7
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Convolution Layer - 8
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                      padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.05),
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(1024, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.08),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        out = self.layers(x)
        """Linear layer requires the tensor to be flattened"""
        out = out.reshape(out.size(0), -1)
        #print(out.shape)
        out = self.fully_connected(out)

        return out








