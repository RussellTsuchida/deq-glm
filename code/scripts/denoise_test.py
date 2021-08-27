# Internals
from ..lib.deq import (ResNetLayerModified, DEQFixedPoint, ConvNet, DEQGLMConv)
from ..lib.plotters import matplotlib_config

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import (DataLoader, Subset)

# Other
import numpy as np
import matplotlib.pyplot as plt
import sys
import time

MAX_EPOCHS  = 5
CHANNELS_1  = 3
CHANNELS_2  = 3
OUTPUT_DIR  = 'outputs/cifar10/'
SEED        = 0 if len(sys.argv) == 1 else int(sys.argv[1])
BATCH_SIZE  = 100
NOISE_STD   = 0.2
PLOT        = True
NUM_STACK   = 1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
matplotlib_config()
torch.manual_seed(SEED)
np.random.seed(SEED)

################################## Load CIFAR10
cifar10_train = datasets.CIFAR10(".", train=True, download=True, transform=transforms.ToTensor())
cifar10_test = datasets.CIFAR10(".", train=False, download=True, transform=transforms.ToTensor())

"""
cifar10_train = Subset(cifar10_train,
    np.random.choice(np.arange(50000),25000,replace=False))
cifar10_test = Subset(cifar10_test, 
    np.random.choice(np.arange(10000),5000,replace=False))
"""

train_loader = DataLoader(cifar10_train, batch_size = BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(cifar10_test, batch_size = BATCH_SIZE, shuffle=False, num_workers=4)

noise = NOISE_STD * torch.randn( [len(cifar10_train)]+ list(cifar10_train[0][0].size())).to(device)
noise = torch.tile(noise, [1,NUM_STACK,1,1])

################################## One training or testing iteration
def epoch(loader, model, opt=None, lr_scheduler=None, plot=False):
    total_loss, total_err = 0.,0.
    model.eval() if opt is None else model.train()
    batch_num = 0
    for X,y in loader:
        X = X.to(device)
        X = torch.tile(X, [1,NUM_STACK,1,1])

        X_noise = X + noise[batch_num*BATCH_SIZE:(batch_num+1)*BATCH_SIZE,:,:,:]
        X_noise = torch.clip(X_noise, 0, 1).to(device)

        Xp = model(X_noise)
        loss = nn.MSELoss()(Xp,X)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            if not (lr_scheduler is None):
                lr_scheduler.step()

        total_loss += loss.item() * X.shape[0]

        if PLOT and not (plot is False) and (batch_num == 0):
            fig, axs = plt.subplots(1, 3)
            
            def custom_imshow(axs, im):
                im = im.swapaxes(0, 1)
                im = im.swapaxes(1, 2)
                im = im.numpy()
                im = im - np.amin(im)
                im = im / np.amax(im)
                axs.imshow(im)

            custom_imshow(axs[0], X[1,0:3,:,:].cpu())
            custom_imshow(axs[1], X_noise[1,0:3,:,:].cpu())
            custom_imshow(axs[2], Xp[1,0:3,:,:].detach().cpu())
            plt.savefig(OUTPUT_DIR  + plot + '_images.pdf')
            plt.close()

        batch_num = batch_num + 1
    
    return total_loss / len(loader.dataset)

experiment_data = np.zeros((6, MAX_EPOCHS+2, 7))
init_types = ['informed', 'random']
init_scale_random = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, None]
init_scale_glm = [10e-4, 10e-3, 10e-2, 0.5, 10e-1, 10e0, 10e1]

for m_idx, initialise_as_glm in enumerate(init_types):
    start = time.time()
    ################################## Initialise the Model
    if initialise_as_glm == 'informed':
        init_scales = init_scale_glm
    else:
        init_scales = init_scale_random

    for init_idx, init_scale in enumerate(init_scales):
        model = DEQGLMConv(3*NUM_STACK, 3, init_type=initialise_as_glm, input_dim=(32,32),
            init_scale = init_scale).to(device)

        ################################## Optimise, minimising the cross entropy loss
        opt = optim.Adam(model.parameters(), lr=1e-3)
        print("# Parmeters: ", sum(a.numel() for a in model.parameters()))

        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, (MAX_EPOCHS+1)*len(train_loader), eta_min=1e-6)
        
        train_err = np.zeros((MAX_EPOCHS+1,))
        test_err = np.zeros((MAX_EPOCHS+1,))
        print('EPOCH|TRAIN_ERR|TEST_ERR')
        print('------------------------')
        for i in range(MAX_EPOCHS+1):
            if i == 0:
                train_err[i] = epoch(train_loader, model)
                plot = initialise_as_glm + '_0'
            else:
                train_err[i] = epoch(train_loader, model, opt, scheduler)
                plot = False
            if i == MAX_EPOCHS:
                plot = initialise_as_glm + '_' + str(i)
            test_err[i] = epoch(test_loader, model, plot = plot)
            print('  ' + str(i).zfill(2) + ' |   ' + str(train_err[i]) + '  |  ' + str(test_err[i]))
        
        print('Took ' + str(time.time() - start) + ' seconds.')

        experiment_data[2*m_idx,    1:, init_idx] = train_err
        experiment_data[2*m_idx+1,  1:, init_idx] = test_err
        experiment_data[2*m_idx,    0, init_idx] = model.spec_norm
        experiment_data[2*m_idx+1,  0, init_idx] = model.spec_norm
        
np.save(OUTPUT_DIR + str(SEED).zfill(4) + '.csv', experiment_data)

