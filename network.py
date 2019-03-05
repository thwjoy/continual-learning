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


def train(args, model, device):
    mnistmTrainSet = ds.mnistmTrainingDataset(
                        text_file=args.dataset_list)

    mnistmTrainLoader = torch.utils.data.DataLoader(
                                            mnistmTrainSet,
                                            batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)
    # start from scratch each time
    epoch = 0

    print('######### Running ckpt: %s ##########' %(args.ckpt))

    while epoch < args.epochs:

        fisher_batch = []

        for i, sample_batched in enumerate(mnistmTrainLoader, 0):
            
            model.train_batch(sample_batched)

            # append some random images to this batch
            fisher_batch.append(sample_batched['image'][0])

            # print statistics
            if i % 50 == 0:
                count = int(model.epoch * math.floor(len(mnistmTrainSet) / (args.batch_size * 200)) + (i / 200))
                print('[%d, %5d] loss: %.3f' %
                    (model.epoch + 1, i + 1, model.loss.item()))
                log_value('loss', model.loss.item(), count)
                _, ind = model.output.max(1)
                name = 'pred_' + str(ind[0])
                sample_image = sample_batched['image'][0]
                log_images(name, sample_image, count)

        model.add_fisher(torch.stack(fisher_batch))
        epoch = epoch + 1
    # update fisher info after each task
    fisher_batch = [iter(mnistmTrainLoader).next()['image'] for i in range(5)]
    fisher_batch = torch.cat(fisher_batch, 0)
    model.update_fisher(fisher_batch)
    

def test(args, model, device):

    mnistmTestSet = ds.mnistTestingDataset(
                        task_list=args.evaluate_list)

    mnistmTestLoader = torch.utils.data.DataLoader(
                                            mnistmTestSet,
                                            batch_size=args.batch_size,
                                            shuffle=True, num_workers=2)

    correct = torch.ByteTensor().to(device)
    
    for i, sample_batched in enumerate(mnistmTestLoader, 0):
        batch_correct = model.test_batch(sample_batched)
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
        batch_correct = model.test_batch(sample_batched)
        correct = torch.cat((correct, batch_correct))

    print('Accuracy on THIS tasks %f' %(correct.float().mean().item()))
        
