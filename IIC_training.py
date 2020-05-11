import pickle
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset

import matplotlib.pyplot as plt
import numpy as np
import os

import code.archs as archs


original_path = os.getcwd()
print(original_path)
config_path = original_path + "/code/mnist_original/config.pickle"
net_path = original_path + "/code/mnist_original/best_net.pytorch"

config_in = open(config_path, "rb")
config = pickle.load(config_in)
net = archs.__dict__[config.arch](config)
net.load_state_dict(torch.load(net_path))

config_dict = config.__dict__
number_of_epochs = len(config_dict['epoch_acc'])
mappings = {pair[0]: pair[1] for pair in config.__dict__['epoch_stats'][number_of_epochs - 1]["best_train_sub_head_match"]}
inv_mappings = {pair[1]: pair[0] for pair in config.__dict__['epoch_stats'][number_of_epochs - 1]["best_train_sub_head_match"]}
best_head = config.__dict__['epoch_stats'][number_of_epochs - 1]["best_train_sub_head"]

print("Epochs", number_of_epochs)
print("Average", config.__dict__['epoch_stats'][number_of_epochs - 1]["avg"])
print("Best mapping", config.__dict__['epoch_stats'][number_of_epochs - 1]["best_train_sub_head_match"])
print("Best head", config.__dict__['epoch_stats'][number_of_epochs - 1]["best_train_sub_head"])