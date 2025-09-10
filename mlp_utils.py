import torch
import torch.nn as nn
from torch.nn import BCELoss
import torch.nn.functional as F
import openslide

from os.path import join 

import joblib
import pandas as pd 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from pytorch_metric_learning.distances import SNRDistance
from pytorch_metric_learning.utils.inference import CustomKNN

import seaborn as sns 

import logging

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from cycler import cycler

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from skimage import io
from skimage.metrics import structural_similarity as ssim
import os
from os import listdir 
from os.path import join 
from pandas import read_pickle, DataFrame, read_csv
import numpy as np 
import pandas as pd 
import openslide 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn import svm
from sklearn.cluster import KMeans
import torch 
from sklearn.semi_supervised import LabelSpreading
import hdbscan
import random
import seaborn as sns 
from collections import defaultdict
import joblib 

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(input_dim))

    def forward(self, x):
        # Apply softmax to attention weights
        weights = F.softmax(self.attention_weights, dim=0)
        # Multiply input by attention weights
        x = x * weights
        return x

class MLPWithAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPWithAttention, self).__init__()
        self.attention = AttentionLayer(input_dim)
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.attention(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.targets = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        return sample.to(device), target.to(device)