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

if not args.dataset in ["MNIST", "ColoredMNIST"]:
    raise Exception("Only MNIST, ColoredMNIST are currently supported.")

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

model_ft = models.resnet18(pretrained=True)
model_ft.eval()
feature_extractor = torch.nn.Sequential(*list(model_ft.children())[:-1])

#feature_extractor.conv1 = Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(3, 3), bias=False)

if use_cuda:
    feature_extractor = feature_extractor.cuda()

transform = transforms.Compose([transforms.Resize((32, 32)),
                                       transforms.ToTensor()])
train_idx = []
test_idx = []
print("Loading dataset...")

#----------------------------------------------------------------------------------

def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr,np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate([np.zeros((h, w, 1), dtype=dtype),arr,np.zeros((h, w, 1), dtype=dtype)], axis=2)
    return arr


class ColoredMNIST(datasets.VisionDataset):
    """
    Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

    Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train', test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
    """
    def __init__(self, root='./data', env='train', transform=None, target_transform=None):
        super(ColoredMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)

        self.prepare_colored_mnist()
        if env in ['train', 'test']:
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
        elif env == 'all_train':
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train.pt')) 
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train, test, and all_train')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target_color, target_digit = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target_color = self.target_transform(target_color)
            target_digit = self.target_transform(target_digit)

        return img, target_color, target_digit

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self):
        colored_mnist_dir = os.path.join(self.root, 'ColoredMNIST')
        if os.path.exists(os.path.join(colored_mnist_dir, 'train.pt')) \
            and os.path.exists(os.path.join(colored_mnist_dir, 'test.pt')):
            print('Colored MNIST dataset already exists')
            return

        print('Preparing Colored MNIST')
        train_mnist = datasets.mnist.MNIST(self.root, train=True, download=True)
        test_mnist = datasets.mnist.MNIST(self.root, train=False, download=True)

        train_set = []
        for idx, (im, label) in enumerate(train_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(train_mnist)}')
            im_array = np.array(im)

            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1

            # Flip label with 25% probability
            if np.random.uniform() < 0.25:
                binary_label = binary_label ^ 1

            # Color the image either red or green according to its possibly flipped label
            color_red = binary_label == 0

            # Flip the color with a probability e that depends on the environment
            # 40% in the first training environment
            if np.random.uniform() < 0.4:
                color_red = not color_red

            colored_arr = color_grayscale_arr(im_array, red=color_red)
            train_set.append((Image.fromarray(colored_arr), binary_label, label))

        
        test_set = []
        for idx, (im, label) in enumerate(test_mnist):
            if idx % 10000 == 0:
                print(f'Converting image {idx}/{len(test_mnist)}')
            im_array = np.array(im)

            # Assign a binary label y to the image based on the digit
            binary_label = 0 if label < 5 else 1

            # Flip label with 25% probability
            if np.random.uniform() < 0.25:
                binary_label = binary_label ^ 1

            # Color the image either red or green according to its possibly flipped label
            color_red = binary_label == 0

            # Flip the color with a probability e that depends on the environment
            # 90% in the test environment
            if np.random.uniform() < 0.9:
                color_red = not color_red

            colored_arr = color_grayscale_arr(im_array, red=color_red)
            test_set.append((Image.fromarray(colored_arr), binary_label, label))


            # Debug
            # print('original label', type(label), label)
            # print('binary label', binary_label)
            # print('assigned color', 'red' if color_red else 'green')
            # plt.imshow(colored_arr)
            # plt.show()
            # break

        if not os.path.exists(colored_mnist_dir):
            os.makedirs(colored_mnist_dir)

        torch.save(train_set, os.path.join(colored_mnist_dir, 'train.pt'))
        torch.save(test_set, os.path.join(colored_mnist_dir, 'test.pt'))


#----------------------------------------------------------------------------------



if args.dataset == "MNIST":
    MNIST_train = datasets.MNIST('./datasets', train=True, download=True, transform=transform)
    MNIST_test = datasets.MNIST('./datasets', train=False, download=True, transform=transform)
    MNIST_train_loader = torch.utils.data.DataLoader(MNIST_train, batch_size=len(MNIST_train), shuffle=False)
    MNIST_test_loader = torch.utils.data.DataLoader(MNIST_test, batch_size=len(MNIST_test), shuffle=False)
    raw_train_data, label_train = next(iter(MNIST_train_loader))  
    raw_test_data, label_test = next(iter(MNIST_test_loader))
    train_idx = list(range(len(MNIST_train)))
    test_idx = list(range(len(MNIST_test)))
elif args.dataset == "ColoredMNIST":
    ColoredMNIST_train = ColoredMNIST(root='./datasets', env='train', transform=transform)
    ColoredMNIST_test = ColoredMNIST(root='./datasets', env='test', transform=transform)
    ColoredMNIST_train_loader = torch.utils.data.DataLoader(ColoredMNIST_train,batch_size=1, shuffle=False)
    ColoredMNIST_test_loader = torch.utils.data.DataLoader(ColoredMNIST_test,batch_size=1, shuffle=False)


with torch.no_grad():
    train_data_list = []
    label_train_color = []
    label_train_digit = []
    for idx, X in tqdm(enumerate(ColoredMNIST_train_loader)):
        data, target_color, target_digit = X
        data = data.cuda()
        feat = feature_extractor(data)
        train_data_list.append(feat)
        label_train_color.append(target_color)
        label_train_digit.append(target_digit)

    test_data_list = []
    label_test_color = []
    label_test_digit = []
    for idx, X in tqdm(enumerate(ColoredMNIST_test_loader)):
        data, target_color, target_digit = X
        data = data.cuda()
        feat = feature_extractor(data)
        test_data_list.append(feat)
        label_test_color.append(target_color)
        label_test_digit.append(target_digit)

data = torch.cat(train_data_list + test_data_list)

label_color = torch.cat(label_train_color + label_test_color)
label_digit = torch.cat(label_train_digit + label_test_digit)

print("Dataset Size: ", label_color.shape[0]) # Should be 190998 according to ESC paper

print('Data preprocessing....')
n_sample = data.shape[0]

print(data.shape)

# scattering transform normalization
data = data.cpu().numpy().reshape(n_sample, -1)

# dimension reduction
data = dim_reduction(data, 500)  # dimension reduction by PCA

label_color = label_color.numpy()
label_digit = label_digit.numpy()          
train_size = len(label_train_color)
test_size = len(label_test_color)

with open('./{}_train_data.pkl'.format(args.dataset), 'wb') as f:
    pickle.dump(data[:train_size], f)
with open('./{}_test_data.pkl'.format(args.dataset), 'wb') as f:
    pickle.dump(data[train_size:], f)
with open('./{}_train_label_color.pkl'.format(args.dataset), 'wb') as f:
    pickle.dump(label_color[:train_size], f)
with open('./{}_test_label_color.pkl'.format(args.dataset), 'wb') as f:
    pickle.dump(label_color[train_size:], f)
with open('./{}_train_label_digit.pkl'.format(args.dataset), 'wb') as f:
    pickle.dump(label_digit[:train_size], f)
with open('./{}_test_label_digit.pkl'.format(args.dataset), 'wb') as f:
    pickle.dump(label_digit[train_size:], f)

print("Done.")
