# Internals
from ..lib.deq import DEQGLMConv
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
from ..lib.data import HyperSpectralData
import math

MAX_EPOCHS  = 5
OUTPUT_DIR  = 'outputs/cifar10/13chan/'
SEED        = 0 if len(sys.argv) == 1 else int(sys.argv[1])
BATCH_SIZE  = 10
NOISE_STD   = 0.2
PLOT        = False
SPEC_START  = -2
SPEC_STOP   = 1
SPEC_NUM    = 25
FILTER_SIZE = 5
NUM_CHANNELS= 13
DATASET     = 'hsi' # 'cifar'

device_idx = SEED % torch.cuda.device_count()
print(device_idx)
device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)

################################## Load CIFAR10
print("initialising data...")
if DATASET == 'cifar':
    cifar10_train = datasets.CIFAR10(".", train=True, download=True, transform=transforms.ToTensor())
    cifar10_test = datasets.CIFAR10(".", train=False, download=True, transform=transforms.ToTensor())
else:
    cifar10_train = HyperSpectralData("/scratch1/tsu007/hsi_road/images/", transforms.ToTensor(), num_channels=NUM_CHANNELS)
    cifar10_train, cifar10_test = torch.utils.data.random_split(cifar10_train,
        [math.floor(6/7*len(cifar10_train)), math.ceil(1/7*len(cifar10_train))],
        generator=torch.Generator().manual_seed(SEED))

#cifar10_train = Subset(cifar10_train,
#    np.random.choice(np.arange(2000),500,replace=False))
#cifar10_test = Subset(cifar10_test, 
#    np.random.choice(np.arange(2000),200,replace=False))

train_loader = DataLoader(cifar10_train, batch_size = BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(cifar10_test, batch_size = BATCH_SIZE, shuffle=False, num_workers=4)

print('initialising noise...')
noise_train = NOISE_STD * torch.randn( [len(cifar10_train)]+ list(cifar10_train[0][0].size()))#.to(device)
noise_test = NOISE_STD * torch.randn( [len(cifar10_test)]+ list(cifar10_test[0][0].size()))#.to(device)

################################## One training or testing iteration
def epoch(loader, model, opt=None, lr_scheduler=None, plot=False, noise=None):
    total_loss, total_err = 0.,0.
    model.eval() if opt is None else model.train()
    batch_num = 0
    for X,y in loader:
        #X = X#.to(device)
        batch_noise = noise[batch_num*BATCH_SIZE:(batch_num+1)*BATCH_SIZE,:,:,:]#.to(device)
        if batch_noise.shape[0] > X.shape[0]:
            batch_noise = batch_noise[:X.shape[0],:,:,:]

        X_noise = X + batch_noise
        X_noise = torch.clip(X_noise, 0, 1).to(device)

        Xp = model(X_noise)
        loss = nn.MSELoss()(Xp,X.to(device))
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


################################################# The actual training loop
init_types = ['informed', 'random']
init_scales = np.logspace(SPEC_START, SPEC_STOP, num=SPEC_NUM)
init_scales = np.concatenate([init_scales,
    np.logspace(-0.1, 0.1, num=SPEC_NUM)])
experiment_data = np.zeros((2*len(init_types), MAX_EPOCHS+2, 
    init_scales.shape[0]))

imsize = (list(cifar10_train[0][0].size())[1], list(cifar10_train[0][0].size())[2])
num_filters = list(cifar10_train[0][0].size())[0]

print("initialising model...")
model = DEQGLMConv(num_filters, FILTER_SIZE, init_type=init_types[0], 
        input_dim=imsize,
        init_scale = init_scales[0], num_hidden=None).to(device)

print("training...")
for m_idx, init_type in enumerate(init_types):
    for init_idx, init_scale in enumerate(init_scales):
        start = time.time()

        ################################## Initialise the Model
        if init_type == 'random':
            init_scale = init_scale/(10*1.9)
        model.init_params(init_type, imsize, init_scale, seed=SEED)
        model.to(device)

        ################################## Optimise, minimising the L2 loss
        opt = optim.Adam(model.parameters(), lr=1e-3)
        print("# Parmeters: ", sum(a.numel() for a in model.parameters()))

        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, (MAX_EPOCHS+1)*len(train_loader), eta_min=1e-6)
        #scheduler = None
        
        train_err = np.zeros((MAX_EPOCHS+1,))
        test_err = np.zeros((MAX_EPOCHS+1,))
        print('EPOCH|TRAIN_ERR|TEST_ERR')
        print('------------------------')
        for i in range(MAX_EPOCHS+1):
            if i == 0:
                train_err[i] = epoch(train_loader, model, noise=noise_train)
                plot = init_type + '_0'
            else:
                train_err[i] = epoch(train_loader, model, opt, scheduler,
                    noise=noise_train)
                plot = False
            if i == MAX_EPOCHS:
                plot = init_type + '_' + str(i)
            test_err[i] = epoch(test_loader, model, plot = plot,
                noise=noise_test)
            print('  ' + str(i).zfill(2) + ' |   ' + str(train_err[i]) + '  |  ' + str(test_err[i]))
        
        print('Took ' + str(time.time() - start) + ' seconds.')

        experiment_data[2*m_idx,    1:, init_idx] = train_err
        experiment_data[2*m_idx+1,  1:, init_idx] = test_err
        experiment_data[2*m_idx,    0, init_idx] = model.spec_norm
        experiment_data[2*m_idx+1,  0, init_idx] = model.spec_norm
        
np.save(OUTPUT_DIR + str(SEED).zfill(4), experiment_data)

