import torch
import torch.utils.data
import os
from skimage import io
import pandas as pd
import numpy as np


class mnistmTrainingDataset(torch.utils.data.Dataset):

    def __init__(self, text_file):
        """
        Args:
            text_file(string): path to text file
        """
        self.name_frame = pd.read_csv(text_file, sep=",", usecols=range(1), header=None)
        self.label_frame = pd.read_csv(text_file, sep=",", usecols=range(1, 2), header=None)

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        img_name = self.name_frame.iloc[idx, 0]
        image = io.imread(img_name).astype(float)
        image *= 1.0/image.max()
        image = np.expand_dims(image, axis=0)
        labels = self.label_frame.iloc[idx, 0]
        sample = {'image': image, 'labels': labels}
        return sample


class mnistTestingDataset(torch.utils.data.Dataset):

    def __init__(self, task_list):
        """
        Args:
            text_file(string): path to text file
        """
        list_names = []
        list_labels = []
        for t in task_list:
            df_name = pd.read_csv(t, sep=",", usecols=range(1), header=None)
            df_list = pd.read_csv(t, sep=",", usecols=range(1, 2), header=None)
            list_names.append(df_name)
            list_labels.append(df_list)

        self.name_frame = pd.concat(list_names, axis = 0, ignore_index = True)
        self.label_frame = pd.concat(list_labels, axis = 0, ignore_index = True)

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        img_name = self.name_frame.iloc[idx, 0]
        image = io.imread(img_name).astype(float)
        image *= 1.0/image.max()
        image = np.expand_dims(image, axis=0)
        labels = self.label_frame.iloc[idx, 0]
        sample = {'image': image, 'labels': labels}
        return sample
