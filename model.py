import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch_utils import torch_io as tio



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



class Model():

    def __init__(self, model_location, device):
        self.model_location = model_location
        self.net = Net()
        self.device = device
        self.net.to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        epoch = [0]
        tio.load_model(model=self.net, optimizer=self.optimizer, epoch=epoch, path=model_location)
        self.epoch = epoch[0]
        

    def train_batch(self, sample_batch):
        input_batch = f.pad(sample_batch['image'].float(), (2, 2, 2, 2))
        input_batch = input_batch.to(self.device)
        self.optimizer.zero_grad()
        self.output = self.net(input_batch)
        self.loss = self.loss_fn(self.output, sample_batch['labels'].to(self.device))
        self.loss.backward()
        self.optimizer.step()


    def save_model(self, model_location):
        tio.save_model(epoch=self.epoch, model=self.net, optimizer=self.optimizer, path=model_location)