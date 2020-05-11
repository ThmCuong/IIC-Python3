import pickle
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset

import matplotlib.pyplot as plt
import numpy as np
import os

import code.archs as archs

import subprocess
# !export CUDA_VISIBLE_DEVICES=0 && python -m code.scripts.cluster.cluster_greyscale_twohead --out_root /content/out/adv --model_ind 687 --arch ClusterNet6cTwoHead --mode IID --dataset_root utils/cluster/MNIST.py --dataset MNIST-uniform-noise --gt_k 10 --output_k_A 10 --output_k_B 10 --lamb_A 1.0 --lamb_B 1.0 --lr 0.0001 --num_epochs 20 --batch_sz 700 --num_dataloaders 5 --num_sub_heads 5 --crop_orig --crop_other --tf1_crop centre_half --tf2_crop random --tf1_crop_sz 20 --tf2_crop_szs 16 20 24 --input_sz 24 --rot_val 25 --no_flip --head_B_epochs 2 --nu 2 --adv_path /content/pgd-adversarials.txt --adv_n 1300

# original_path = os.getcwd()
# print(original_path)
# config_path = original_path + "/code/mnist_original/config.pickle"
# net_path = original_path + "/code/mnist_original/best_net.pytorch"

# config_in = open(config_path, "rb")
# config = pickle.load(config_in)
# net = archs.__dict__[config.arch](config)
# net.load_state_dict(torch.load(net_path))

# config_dict = config.__dict__
# number_of_epochs = len(config_dict['epoch_acc'])
# mappings = {pair[0]: pair[1] for pair in config.__dict__['epoch_stats'][number_of_epochs - 1]["best_train_sub_head_match"]}
# inv_mappings = {pair[1]: pair[0] for pair in config.__dict__['epoch_stats'][number_of_epochs - 1]["best_train_sub_head_match"]}
# best_head = config.__dict__['epoch_stats'][number_of_epochs - 1]["best_train_sub_head"]

# print("Epochs", number_of_epochs)
# print("Average", config.__dict__['epoch_stats'][number_of_epochs - 1]["avg"])
# print("Best mapping", config.__dict__['epoch_stats'][number_of_epochs - 1]["best_train_sub_head_match"])
# print("Best head", config.__dict__['epoch_stats'][number_of_epochs - 1]["best_train_sub_head"])