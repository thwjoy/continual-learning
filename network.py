import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
from tensorboard_logger import log_value, log_images
import os
import sys
import math
from torch_utils import dataset as ds
from torch_utils import torch_io as tio
import matplotlib.pyplot as plt


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


def train(args):

    # tensorboard
    run_name = "./runs/run-classifier_batch_" + str(args.batch_size) \
                    + "_epochs_" + str(args.epochs) + "_" + args.log_message

    print('######### Run Name: %s ##########' %(run_name))

    net = Net()

    mnistmTrainSet = ds.mnistmTrainingDataset(
                        text_file=args.dataset_list)

    mnistmTrainLoader = torch.utils.data.DataLoader(
                                            mnistmTrainSet,
                                            batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # put on gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)

    epoch = [0]

    # load prev model
    tio.load_model(model=net, optimizer=optimizer, epoch=epoch, path=run_name + '/ckpt')
    epoch = 0
    print('######### Loaded ckpt: %s ##########' %(run_name))

    while epoch < args.epochs:

        for i, sample_batched in enumerate(mnistmTrainLoader, 0):
            input_batch = f.pad(sample_batched['image'].float(), (2, 2, 2, 2))
            input_batch = input_batch.to(device)

            optimizer.zero_grad()

            output = net(input_batch)
            loss = loss_fn(output, sample_batched['labels'].to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            if i % 50 == 0:
                count = int(epoch * math.floor(len(mnistmTrainSet) / (args.batch_size * 200)) + (i / 200))
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, loss.item()))
                log_value('loss', loss.item(), count)
                _, ind = output.max(1)
                name = 'pred_' + str(ind[0])
                sample_image = sample_batched['image'][0]
                log_images(name, sample_image, count)

        # save model
        tio.save_model(epoch=epoch, model=net, optimizer=optimizer, path=run_name + '/ckpt')
        epoch = epoch + 1

def test(args):


    if not args.ckpt:
        ckpt = "./runs/run-classifier_batch_" + str(args.batch_size) \
                    + "_epochs_" + str(args.epochs) + "_" + args.log_message + '/ckpt'
    else:
        ckpt = args.ckpt

    net = Net()

    mnistmTestSet = ds.mnistTestingDataset(
                        task_list=args.evaluate_list)

    mnistmTestLoader = torch.utils.data.DataLoader(
                                            mnistmTestSet,
                                            batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)
    # put on gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device)
 
    # load prev model
    tio.load_test_model(model=net, path=ckpt)

    correct = torch.ByteTensor().to(device)
    
    for i, sample_batched in enumerate(mnistmTestLoader, 0):
        input_batch = f.pad(sample_batched['image'].float(), (2, 2, 2, 2), mode='constant', value=0)
        input_batch = input_batch.to(device)
        labels = sample_batched['labels']

        output = net(input_batch)

        batch_correct = (torch.argmax(output, dim=1) == labels.to(device))
        correct = torch.cat((correct, batch_correct))

    print('Accuracy on ALL tasks %f' %(correct.float().mean().item()))

    ##### Show accuracy on only this task
    mnistmTestSet = ds.mnistmTrainingDataset(
                        text_file=args.evaluate_list[-1])

    mnistmTestLoader = torch.utils.data.DataLoader(
                                            mnistmTestSet,
                                            batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    correct = torch.ByteTensor().to(device)
    
    for i, sample_batched in enumerate(mnistmTestLoader, 0):
        input_batch = f.pad(sample_batched['image'].float(), (2, 2, 2, 2), mode='constant', value=0)
        input_batch = input_batch.to(device)
        labels = sample_batched['labels']

        output = net(input_batch)

        batch_correct = (torch.argmax(output, dim=1) == labels.to(device))
        correct = torch.cat((correct, batch_correct))


    print('Accuracy on THIS tasks %f' %(correct.float().mean().item()))
        
