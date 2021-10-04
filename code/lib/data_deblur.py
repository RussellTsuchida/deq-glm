# Blur torchvision Dataset

from torch.utils.data import TensorDataset
import torch

import cv2
import numpy as np
import sys

# Set random seed
# SEED = 0
# torch.manual_seed(SEED)
# np.random.seed(SEED)


def dataset_blur(dataset, min_sigma=0, max_sigma=3, normalise=True):
    """Return numpy array of blurred torchvision dataset (NxHxWxC) by applying random Gaussian noise"""
    
    blurred = np.zeros_like(dataset.data)
    for i in range(dataset.data.shape[0]):
        blurred[i] = cv2.GaussianBlur(src=dataset.data[i], ksize=(0,0), sigmaX=np.random.uniform(min_sigma, max_sigma))
    
    if normalise:
        blurred = blurred/255.
    
    return blurred


def traintest_blur(train_dataset, test_dataset, min_sigma=0, max_sigma=3):
    """Return train and test normalised blurred TensorDataset (NxCxHxW)"""
    
    train_blur = dataset_blur(train_dataset, min_sigma, max_sigma, normalise=True)
    test_blur = dataset_blur(test_dataset, min_sigma, max_sigma, normalise=True)
    
    train_dataset = TensorDataset(torch.Tensor(train_blur).permute(0,3,1,2),torch.Tensor(train_dataset.data/255.).permute(0,3,1,2))
    test_dataset = TensorDataset(torch.Tensor(test_blur).permute(0,3,1,2),torch.Tensor(test_dataset.data/255.).permute(0,3,1,2))
    
    return train_dataset, test_dataset

