import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch_utils import torch_io as tio
from torch import autograd



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


def get_second_order_grad(grads, xs):
    grads2 = []
    for j, (grad, x) in enumerate(zip(grads, xs)):
        print('2nd order on ', j, 'th layer')
        print(x.size())
        grad = torch.reshape(grad, [-1])
        grads2_tmp = []
        for count, g in enumerate(grad):
            g2 = torch.autograd.grad(g, x, retain_graph=True)[0]
            g2 = torch.reshape(g2, [-1])
            grads2_tmp.append(g2[count].data.cpu().numpy())
        grads2.append(torch.from_numpy(np.reshape(grads2_tmp, x.size())).to(DEVICE_IDS[0]))
    for grad in grads2:  # check size
        print(grad.size())

    return grads2


class Model():

    def __init__(self, model_location, device, monte_carlo_sample_size=200):
        self.model_location = model_location
        self.net = Net()
        self.device = device
        self.net.to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        epoch = [0]
        tio.load_model(model=self.net, optimizer=self.optimizer, epoch=epoch, path=model_location)
        self.epoch = epoch[0]
        
    def add_fisher(self, input_batch):
        # we need to sample some examples and then perform softmax over them, 
        # we use this to get the current probability distribution
        input_batch = f.pad(input_batch.float(), (2, 2, 2, 2))
        input_batch = input_batch.to(self.device)
        outputs = self.net(input_batch)
        
        # get log probabilities
        log_softmax = torch.nn.functional.log_softmax(outputs, 1)

        # get gradients wrt parameters

        print(log_softmax)

        log_grads = autograd.grad(log_softmax[0], self.net.parameters(), retain_graph=True)

        print(log_grads)
        # self.fisher = torch.(log_grads, log_grads)


        
        # this then becomes the emperical fisher matrix

    # def ewc_loss(self, params):
    #     """
    #     The regulariser for EWC
    #     """
    #     # get the sum of previous fisher matrices from prev task, then just do weighted least squares

    #     return


    def train_batch(self, sample_batch):
        input_batch = f.pad(sample_batch['image'].float(), (2, 2, 2, 2))
        input_batch = input_batch.to(self.device)
        self.optimizer.zero_grad()
        self.output = self.net(input_batch)
        self.loss = self.loss_fn(self.output, sample_batch['labels'].to(self.device))
        self.loss.backward()
        self.optimizer.step()

    def test_batch(self, sample_batch):
        input_batch = f.pad(sample_batch['image'].float(), (2, 2, 2, 2), mode='constant', value=0)
        input_batch = input_batch.to(self.device)
        labels = sample_batch['labels']
        output = self.net(input_batch)
        batch_correct = (torch.argmax(output, dim=1) == labels.to(self.device))
        return batch_correct


    def save_model(self, model_location):
        tio.save_model(epoch=self.epoch, model=self.net, optimizer=self.optimizer, path=model_location)