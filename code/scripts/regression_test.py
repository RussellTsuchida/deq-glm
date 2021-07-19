# Internals
from ..lib.deq import (FullyConnectedLayer, DEQFixedPoint)
from ..lib.data import SequenceOneDimension

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
DIMENSION_Y = 500
NOISE_VAR   = 0.01

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

################################## Initialise the Model
f = FullyConnectedLayer(DIMENSION_Y, DIMENSION_Y, DIMENSION_Y)
model = nn.Sequential(DEQFixedPoint(f, solver=None, tol=1e-2, max_iter=25, m=5)).\
                      to(device)

################################## Load regression data
TARGET_FUN  = lambda x: np.exp(-np.abs(x)) * np.sin(x) + np.exp(-(x+9)**2)
OFFSET      = 2

train_data = SequenceOneDimension(-2*np.pi, 2*np.pi, DIMENSION_Y, OFFSET,
        NOISE_VAR)
X_train_input, Y_train_input, X_train_target, Y_train_target = \
        train_data.sample_inputs_targets(NUM_TRAIN, TARGET_FUN)

test_data = SequenceOneDimension(-2*np.pi, 2*np.pi, DIMENSION_Y, OFFSET,
        NOISE_VAR)
X_test_input, Y_test_input, X_test_target, Y_test_target = \
        test_data.sample_inputs_targets(NUM_TEST, TARGET_FUN)

plot_data = SequenceOneDimension(-2*np.pi, 2*np.pi, DIMENSION_Y, OFFSET,
        0)
X_plot_input, Y_plot_input, X_plot_target, Y_plot_target = \
        plot_data.sample_inputs_targets(NUM_PLOT, TARGET_FUN)

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

max_epochs = 20
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

