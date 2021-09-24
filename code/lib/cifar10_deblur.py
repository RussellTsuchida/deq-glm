# Deblur CIFAR10 Dataset
# Usage:
# cifar10_train_dataset, cifar10_test_dataset = cifar10_tensor(cifar10_train, cifar10_test)
# train_loader = DataLoader(cifar10_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# test_loader = DataLoader(cifar10_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

from torch.utils.data import TensorDataset
import torch

import cv2
import numpy as np
import sys

SEED        = 0 if len(sys.argv) == 1 else int(sys.argv[1])

torch.manual_seed(SEED)
np.random.seed(SEED)


def blur_dataset(dataset, min_sigma=0, max_sigma=3, normalise=True):
    """Return torch tensor of blurred dataset by applying random Gaussian noise"""
    
    blurred = np.zeros_like(dataset.data)
    for i in range(dataset.data.shape[0]):
        blurred[i] = cv2.GaussianBlur(src=dataset.data[i], ksize=(0,0), sigmaX=np.random.uniform(min_sigma, max_sigma))
    
    if normalise:
        blurred = blurred/255.
    
    return blurred


def cifar10_tensor(cifar10_train, cifar10_test):
    """Return train and test CIFAR10 TensorDataset to load to DataLoader"""
    
    cifar10_train_blur = blur_dataset(cifar10_train)
    cifar10_test_blur = blur_dataset(cifar10_test)
    cifar10_train_dataset = TensorDataset(torch.Tensor(cifar10_train_blur), torch.Tensor(cifar10_train.data))
    cifar10_test_dataset = TensorDataset(torch.Tensor(cifar10_test_blur), torch.Tensor(cifar10_test.data))
    return cifar10_train_dataset, cifar10_test_dataset

