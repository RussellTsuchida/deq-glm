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
NUM_TEST    = 10000
NUM_PLOT    = 100
DIMENSION_Y = 100
NOISE_VAR   = 0.01
MAX_EPOCHS  = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

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

markers = ['-', '--']
error_fig = plt.figure()
for m_idx, initialise_as_glm in enumerate([True, False]):
    ################################## Initialise the Model
    if initialise_as_glm:
        x_init = [X_train_input, X_train_target]
        y_init = Y_train_input
        save_dir = 'glm_init/'
    else:
        x_init = None
        y_init = None
        save_dir = 'not_glm_init/'
    f = FullyConnectedLayer(DIMENSION_Y, DIMENSION_Y, DIMENSION_Y, 
            x_init = x_init, y_init = y_init)
    model = nn.Sequential(DEQFixedPoint(f, solver=None, tol=1e-2, max_iter=25, m=5)).\
                          to(device)

    ################################## One training or testing iteration
    def epoch(data, model, opt=None, lr_scheduler=None):
        total_loss = 0.
        model.eval() if opt is None else model.train()

        X, y = data
        yp = model(X)
        loss = nn.MSELoss()(yp,y)
        #loss = nn.L1Loss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
            lr_scheduler.step()
        
        total_loss += loss.item() * list(X.shape)[0]

        return total_loss / list(X.shape)[0]

    ################################## Optimise, minimising the loss
    opt = optim.Adam(model.parameters(), lr=1e-3)
    print("# Parmeters: ", sum(a.numel() for a in model.parameters()))


    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS*NUM_TRAIN, eta_min=1e-6)
    train_err = np.zeros((MAX_EPOCHS,))
    test_err = np.zeros((MAX_EPOCHS,))
    for i in range(MAX_EPOCHS):
        train_err[i] = epoch([Y_train_input, Y_train_target], model, opt, scheduler)
        test_err[i] = epoch([Y_test_input, Y_test_target], model)

    matplotlib_config()

    xlim = [-2*np.pi-OFFSET, 2*np.pi]
    ylim = [-5, 5]
    plot_1d_sequence_data(X_train_input, Y_train_input, 
            X_plot_input, Y_plot_input, save_dir + 'plot_input_train.pdf', xlim, ylim)
    plot_1d_sequence_data(X_train_target, Y_train_target, 
            X_plot_target, Y_plot_target, save_dir + 'plot_target_train.pdf', xlim, ylim)
    pred = model(Y_train_input).detach().numpy()
    plot_1d_sequence_data(X_train_target, pred, 
            X_plot_target, Y_plot_target, save_dir + 'plot_target_train_pred.pdf', xlim, ylim)
    plot_1d_sequence_data(X_test_input, Y_test_input, 
            X_plot_input, Y_plot_input, save_dir + 'plot_input_test.pdf', xlim, ylim)
    plot_1d_sequence_data(X_test_target, Y_test_target, 
            X_plot_target, Y_plot_target, save_dir + 'plot_target_test.pdf', xlim, ylim)
    pred = model(Y_test_input).detach().numpy()
    plot_1d_sequence_data(X_test_target, pred, 
            X_plot_target, Y_plot_target, save_dir + 'plot_target_test_pred.pdf', xlim, ylim)

    # Plot error curves
    plt.figure(error_fig.number)
    plt.plot(np.log(train_err), c='b', label='training', ls=markers[m_idx])
    plt.plot(np.log(test_err), c='r', label='testing', ls=markers[m_idx])
    plt.ylabel('$\log$ MSE')
    plt.xlabel('Training epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig('error_curve.pdf')
