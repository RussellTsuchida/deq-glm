# Internals
from ..lib.deq import (ResNetLayer, DEQFixedPoint, ConvNet)

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Other
import numpy as np
import sys

MAX_EPOCHS  = 50
CHANNELS_1  = 48
CHANNELS_2  = 64
OUTPUT_DIR  = 'outputs/cifar10/'
SEED        = 0 if len(sys.argv) == 1 else int(sys.argv[1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)

################################## Load CIFAR10
cifar10_train = datasets.CIFAR10(".", train=True, download=True, transform=transforms.ToTensor())
cifar10_test = datasets.CIFAR10(".", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(cifar10_train, batch_size = 100, shuffle=True, num_workers=4)
test_loader = DataLoader(cifar10_test, batch_size = 100, shuffle=False, num_workers=4)

################################## One training or testing iteration
def epoch(loader, model, opt=None, lr_scheduler=None):
    total_loss, total_err = 0.,0.
    model.eval() if opt is None else model.train()
    for X,y in loader:
        X,y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]

    return total_err / len(loader.dataset), total_loss / len(loader.dataset)

experiment_data = np.zeros((6, MAX_EPOCHS))
#init_types = ['informed', 'naive', 'random']
init_types = ['random']

for m_idx, initialise_as_glm in enumerate(init_types):
    ################################## Initialise the Model
    #f = ConvNet(CHANNELS_1, 64, kernel_size=3)
    f = ResNetLayer(CHANNELS_1, CHANNELS_2, kernel_size=3, init_as='informed')
    model = nn.Sequential(nn.Conv2d(3,CHANNELS_1, kernel_size=3, bias=True, padding=1),
                          nn.BatchNorm2d(CHANNELS_1),
                          DEQFixedPoint(f, solver=None, tol=1e-2, max_iter=25, m=5),
                          nn.BatchNorm2d(CHANNELS_1),
                          nn.AvgPool2d(8,8),
                          nn.Flatten(),
                          nn.Linear(CHANNELS_1*4*4,10)).to(device)


    ################################## Optimise, minimising the cross entropy loss
    opt = optim.Adam(model.parameters(), lr=1e-3)
    print("# Parmeters: ", sum(a.numel() for a in model.parameters()))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS*len(train_loader), eta_min=1e-6)
    
    train_acc = np.zeros((MAX_EPOCHS,))
    test_acc = np.zeros((MAX_EPOCHS,))
    for i in range(MAX_EPOCHS):
        train_acc[i] = epoch(train_loader, model, opt, scheduler)[0]
        test_acc[i] = epoch(test_loader, model)[0]
        print(train_acc[i])
        print(test_acc[i])

    experiment_data[2*m_idx,:] = train_acc
    experiment_data[2*m_idx+1,:] = test_acc
        
np.savetxt(OUTPUT_DIR + str(SEED).zfill(4) + '.csv', experiment_data)

