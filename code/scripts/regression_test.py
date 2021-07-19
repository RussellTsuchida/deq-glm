# Internals
from ..lib.deq import (FullyConnectedLayer, DEQFixedPoint)
from ..lib.data import SequenceOneDimension
from ..lib.plotters import (matplotlib_gc, matplotlib_config, 
        plot_1d_sequence_data)

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
    total_loss = 0.
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

    return total_loss / list(X.shape)[0]

################################## Optimise, minimising the cross entropy loss
opt = optim.Adam(model.parameters(), lr=1e-3)
print("# Parmeters: ", sum(a.numel() for a in model.parameters()))

max_epochs = 20
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs*NUM_TRAIN, eta_min=1e-6)
train_err = np.zeros((max_epochs,))
test_err = np.zeros((max_epochs,))
for i in range(max_epochs):
    train_err[i] = epoch([Y_train_input, Y_train_target], model, opt, scheduler)
    test_err[i] = epoch([Y_test_input, Y_test_target], model)

matplotlib_config()

xlim = [-2*np.pi-OFFSET, 2*np.pi]
ylim = [-0.6, 0.8]
plot_1d_sequence_data(X_train_input, Y_train_input, 
        X_plot_input, Y_plot_input, 'plot_input_train.pdf', xlim, ylim)
plot_1d_sequence_data(X_train_target, Y_train_target, 
        X_plot_target, Y_plot_target, 'plot_target_train.pdf', xlim, ylim)
pred = model(Y_train_input).detach().numpy()
plot_1d_sequence_data(X_train_target, pred, 
        X_plot_target, Y_plot_target, 'plot_target_train_pred.pdf', xlim, ylim)
plot_1d_sequence_data(X_test_input, Y_test_input, 
        X_plot_input, Y_plot_input, 'plot_input_test.pdf', xlim, ylim)
plot_1d_sequence_data(X_test_target, Y_test_target, 
        X_plot_target, Y_plot_target, 'plot_target_test.pdf', xlim, ylim)
pred = model(Y_test_input).detach().numpy()
plot_1d_sequence_data(X_test_target, pred, 
        X_plot_target, Y_plot_target, 'plot_target_test_pred.pdf', xlim, ylim)

# Plot error curves
plt.plot(train_err, label='training')
plt.plot(test_err, label='training')
plt.legend()
plt.savefig('error_curve.pdf')
plt.close()

