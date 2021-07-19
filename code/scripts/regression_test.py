# Internals
from ..lib.deq import (FullyConnectedLayer, DEQFixedPoint)

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Numpy
import numpy as np
import matplotlib.pyplot as plt


NUM_TRAIN   = 100
NUM_TEST    = 100
NUM_PLOT    = 1000
DIMENSION_Y = 1000
NOISE_VAR   = 0.01

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

################################## Initialise the Model
f = FullyConnectedLayer(DIMENSION_Y, DIMENSION_Y, DIMENSION_Y)
model = nn.Sequential(DEQFixedPoint(f, solver=None, tol=1e-2, max_iter=25, m=5)).\
                      to(device)

################################## Load CIFAR10
TARGET_FUN  = lambda x: np.exp(-np.abs(x)) * np.sin(x) + np.exp(-(x+9)**2)
OFFSET      = 2

X_train_input = np.tile(np.linspace(-2*np.pi, 2*np.pi, DIMENSION_Y),
            [NUM_TRAIN, 1])
X_train_target = np.tile(np.linspace(-2*np.pi-OFFSET, 2*np.pi-OFFSET, DIMENSION_Y),
            [NUM_TRAIN, 1])
Y_train_input = TARGET_FUN(X_train_input) + np.random.normal(0, np.sqrt(NOISE_VAR),
        (NUM_TRAIN, DIMENSION_Y))
Y_train_target = TARGET_FUN(X_train_target) + np.random.normal(0, np.sqrt(NOISE_VAR),
        (NUM_TRAIN, DIMENSION_Y))

X_test_input = np.tile(np.linspace(-2*np.pi, 2*np.pi, DIMENSION_Y),
            [NUM_TEST, 1])
X_test_target = np.tile(np.linspace(-2*np.pi-OFFSET, 2*np.pi-OFFSET, DIMENSION_Y),
            [NUM_TEST, 1])
Y_test_input = TARGET_FUN(X_test_input) + np.random.normal(0, np.sqrt(NOISE_VAR),
        (NUM_TEST, DIMENSION_Y))
Y_test_target = TARGET_FUN(X_test_target) + np.random.normal(0, np.sqrt(NOISE_VAR),
        (NUM_TEST, DIMENSION_Y))

X_plot_input = np.tile(np.linspace(-2*np.pi, 2*np.pi, DIMENSION_Y),
            [NUM_PLOT, 1])
X_plot_target = np.tile(np.linspace(-2*np.pi-OFFSET, 2*np.pi-OFFSET, DIMENSION_Y),
            [NUM_PLOT, 1])
Y_plot_input = TARGET_FUN(X_plot_input) 
Y_plot_target = TARGET_FUN(X_plot_target) 

X_train_input = torch.from_numpy(X_train_input.astype(np.float32))
X_train_target = torch.from_numpy(X_train_target.astype(np.float32))
Y_train_input = torch.from_numpy(Y_train_input.astype(np.float32))
Y_train_target = torch.from_numpy(Y_train_target.astype(np.float32))

X_test_input = torch.from_numpy(X_test_input.astype(np.float32))
X_test_target = torch.from_numpy(X_test_target.astype(np.float32))
Y_test_input = torch.from_numpy(Y_test_input.astype(np.float32))
Y_test_target = torch.from_numpy(Y_test_target.astype(np.float32))

X_plot_input = torch.from_numpy(X_plot_input.astype(np.float32))
X_plot_target = torch.from_numpy(X_plot_target.astype(np.float32))
Y_plot_input = torch.from_numpy(Y_plot_input.astype(np.float32))
Y_plot_target = torch.from_numpy(Y_plot_target.astype(np.float32))


################################## One training or testing iteration
def epoch(data, model, opt=None, lr_scheduler=None):
    total_loss, total_err = 0.,0.
    model.eval() if opt is None else model.train()

    X, y = data
    yp = model(X)
    loss = nn.MSELoss()(yp,y)
    if opt:
        opt.zero_grad()
        loss.backward()
        opt.step()
        lr_scheduler.step()
    
    total_loss += loss.item() * list(X.shape)[0]

    return total_err / list(X.shape)[0], total_loss / list(X.shape)[0]

################################## Optimise, minimising the cross entropy loss
opt = optim.Adam(model.parameters(), lr=1e-3)
print("# Parmeters: ", sum(a.numel() for a in model.parameters()))

max_epochs = 100
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs*NUM_TRAIN, eta_min=1e-6)

for i in range(max_epochs):
    print(epoch([Y_train_input, Y_train_target], model, opt, scheduler))
    print(epoch([Y_test_input, Y_test_target], model))

ALPHA = 0.2

# Plot training input data
for p in range(5):
    plt.plot(X_train_input[p,:], Y_train_input[p,:], 
            label='train_input'+str(p), alpha=ALPHA)
plt.plot(X_plot_input[0,:], Y_plot_input[0,:], lw=2, c='k', label='gt')
plt.legend()
plt.savefig('plot_input_train.pdf')
plt.close()

# Plot training target data
for p in range(5):
    plt.plot(X_train_target[p,:], Y_train_target[p,:], 
            label='train_target'+str(p), alpha=ALPHA)
plt.plot(X_plot_target[0,:], Y_plot_target[0,:], lw=2, c='k', label='gt')
plt.legend()
plt.savefig('plot_target_train.pdf')
plt.close()

# Plot test target data
for p in range(5):
    plt.plot(X_test_target[p,:], Y_test_target[p,:], 
            label='test_target'+str(p), alpha=ALPHA)
plt.plot(X_plot_target[0,:], Y_plot_target[0,:], lw=2, c='k', label='gt')
plt.legend()
plt.savefig('plot_target_test.pdf')
plt.close()

# Plot test prediction data
pred = model(Y_test_input).detach().numpy()
for p in range(5):
    plt.plot(X_test_target[p,:], pred[p,:], 
            label='test_prediction'+str(p), alpha=ALPHA)
plt.plot(X_plot_target[0,:], Y_plot_target[0,:], lw=2, c='k', label='gt')
plt.legend()
plt.savefig('plot_target_pred.pdf')
plt.close()

