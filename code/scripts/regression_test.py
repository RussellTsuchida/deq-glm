# Internals
from ..lib.deq import (FullyConnectedLayer, DEQFixedPoint, DEQGLM)
from ..lib.data import SequenceOneDimension
from ..lib.plotters import (matplotlib_gc, matplotlib_config, 
        plot_1d_sequence_data)
# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Others
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.spatial as sp

NUM_TRAIN   = 20000
NUM_TEST    = 2000
DIMENSION_Y = 100
NOISE_VAR   = 0.1
MAX_EPOCHS  = 100
PLOT        = True
TARGET_FUN  = lambda x: (np.exp(-0.1*np.abs(x)**2)*np.sin(x) \
        + np.exp(-(x+9)**2))
OFFSET      = 2
OUTPUT_DIR  = 'outputs/'
SEED        = 0 if len(sys.argv) == 1 else int(sys.argv[1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

################################## Load regression data
train_data = SequenceOneDimension(-2*np.pi, 2*np.pi, DIMENSION_Y, OFFSET,
        NOISE_VAR)
X_train_input, Y_train_input, X_train_target, Y_train_target = \
        train_data.sample_inputs_targets(NUM_TRAIN, TARGET_FUN,
                normalise_y = True, normalise_x = False)

test_data = SequenceOneDimension(-2*np.pi, 2*np.pi, DIMENSION_Y, OFFSET,
        NOISE_VAR)
X_test_input, Y_test_input, X_test_target, Y_test_target = \
        test_data.sample_inputs_targets(NUM_TEST, TARGET_FUN,
        normalise_y = [train_data.mean_y, train_data.std_y])
#        normalise_x = [train_data.mean_x, train_data.std_x])

plot_data = SequenceOneDimension(-2*np.pi, 2*np.pi, DIMENSION_Y, OFFSET,
        0)
X_plot_input, Y_plot_input, X_plot_target, Y_plot_target = \
        plot_data.sample_inputs_targets(1, TARGET_FUN,
        normalise_y = [train_data.mean_y, train_data.std_y])
#        normalise_x = [train_data.mean_x, train_data.std_x])

markers = ['-', '--', '-.']
error_fig = plt.figure()
# train/test error curves for standard and our init
experiment_data = np.zeros((6, MAX_EPOCHS)) 

for m_idx, initialise_as_glm in enumerate(['informed', 'naive', 'random']):
    ################################## Initialise the Model
    kernel = None
    if initialise_as_glm == 'informed':
        x_init = [X_train_input[0:1,:], X_train_target[0:1,:]]
        save_dir = 'glm_init_informed/'
        kernel = lambda x1, x2: \
                        (np.exp(-sp.distance.cdist(x1, x2, 'sqeuclidean')/2)*\
                         np.exp(-np.sin(sp.distance.cdist(x1, x2, 'euclidean'))**2)).\
                        astype(np.float32)
        """
        kernel = lambda x1, x2: \
                        (np.exp(-sp.distance.cdist(x1, x2, 'sqeuclidean')/2)).\
                        astype(np.float32)
        """
    elif initialise_as_glm == 'naive':
        x_init = True
        save_dir = 'glm_init_naive/'
    else:
        x_init = False
        save_dir = 'not_glm_init/'
    f = FullyConnectedLayer(DIMENSION_Y, DIMENSION_Y, DIMENSION_Y, 
            x_init = x_init, kernel=kernel)
    model = DEQGLM(f, solver=None, tol=1e-2, max_iter=25, m=5)

    ################################## One training or testing iteration
    def epoch(data, model, opt=None):
        total_loss = 0.
        model.eval() if opt is None else model.train()

        X, y = data
        yp = model(X)
        loss = nn.MSELoss()(yp,y)
        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        total_loss += loss.item() * list(X.shape)[0]

        return total_loss / list(X.shape)[0]

    ################################## Optimise, minimising the loss
    opt = optim.Adam(model.parameters(), lr=1e-3)
    print("# Parmeters: ", sum(a.numel() for a in model.parameters()))



    train_err = np.zeros((MAX_EPOCHS,))
    test_err = np.zeros((MAX_EPOCHS,))
    for i in range(MAX_EPOCHS):
        print(i)
        train_err[i] = epoch([Y_train_input, Y_train_target], model, opt)
        test_err[i] = epoch([Y_test_input, Y_test_target], model)
    
    if PLOT:
        matplotlib_config()

        xlim = [-2*np.pi-OFFSET, 2*np.pi]
        ylim = [-2.5, 3.5]
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
        plt.savefig(OUTPUT_DIR + 'error_curve.pdf')

    experiment_data[2*m_idx,:] = train_err
    experiment_data[2*m_idx+1,:] = test_err

np.savetxt(OUTPUT_DIR + str(SEED).zfill(4) + '.csv', experiment_data)
