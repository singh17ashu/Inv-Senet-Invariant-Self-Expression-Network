import os
import numpy as np

import torch
import pickle
import argparse
from tqdm import tqdm

from decomposition.dim_reduction import dim_reduction
from torchvision import datasets, transforms


from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms, datasets, models
import torchvision.datasets.utils as dataset_utils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="MNIST")
args = parser.parse_args()

if not args.dataset in ["MNIST"]:
    raise Exception("Only MNIST are currently supported.")

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

model_ft = models.resnet18(pretrained=True)
model_ft.eval()
feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])

if use_cuda:
    feature_extractor = feature_extractor.cuda()

transform = transforms.Compose([transforms.ToTensor()])
train_idx = []
test_idx = []
print("Loading dataset...")

#----------------------------------------------------------------------------------

if args.dataset == "MNIST":
    MNIST_train = datasets.MNIST('./datasets', train=True, download=True, transform=transform)
    MNIST_test = datasets.MNIST('./datasets', train=False, download=True, transform=transform)
    MNIST_train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=1, shuffle=False)
    MNIST_test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size=1, shuffle=False)

with torch.no_grad():
    train_data_list = []
    label_train_digit = []
    for idx, X in tqdm(enumerate(MNIST_train_loader)):
        data, target_digit = X
        data = data.repeat(1,3,1,1)
        data = data.cuda()
        feat = feature_extractor(data)
        train_data_list.append(feat)
        label_train_digit.append(target_digit)

    test_data_list = []
    label_test_digit = []
    for idx, X in tqdm(enumerate(MNIST_test_loader)):
        data, target_digit = X
        data = data.repeat(1,3,1,1)
        data = data.cuda()
        feat = feature_extractor(data)
        test_data_list.append(feat)
        label_test_digit.append(target_digit)

data = torch.cat(train_data_list + test_data_list)
label_digit = torch.cat(label_train_digit + label_test_digit)

print("Dataset Size: ", label_digit.shape[0]) # Should be 190998 according to ESC paper

print('Data preprocessing....')
n_sample = data.shape[0]

print(data.shape)

# scattering transform normalization
data = data.cpu().numpy().reshape(n_sample, -1)


label_digit = label_digit.numpy()          
train_size = len(label_train_digit)
test_size = len(label_test_digit)

with open('./{}_train_data.pkl'.format(args.dataset), 'wb') as f:
    pickle.dump(data[:train_size], f)
with open('./{}_test_data.pkl'.format(args.dataset), 'wb') as f:
    pickle.dump(data[train_size:], f)
with open('./{}_train_label_digit.pkl'.format(args.dataset), 'wb') as f:
    pickle.dump(label_digit[:train_size], f)
with open('./{}_test_label_digit.pkl'.format(args.dataset), 'wb') as f:
    pickle.dump(label_digit[train_size:], f)

print("Done.")
