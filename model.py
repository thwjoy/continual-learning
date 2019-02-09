import torch
import torch.nn as nn
import torch.nn.functional as f


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # single input channel and 6 output, 5*5 kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool2d = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool2d(f.relu(self.conv1(x)))
        x = self.pool2d(f.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class model():

    def __init__(self, model_location):
        self.model_location = model_location
        self.net = Net()
