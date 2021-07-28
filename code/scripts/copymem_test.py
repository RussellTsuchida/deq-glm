# Internals
from ..lib.deq import (FullyConnectedLayer, DEQFixedPoint, DEQGLM)
from ..lib.data import CopyMemory
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

NUM_TRAIN   = 200
NUM_TEST    = 20
MEM_LENGTH  = 400
NUM_PLOT    = 100
MAX_EPOCHS  = 100
PLOT        = False
OFFSET      = 2
OUTPUT_DIR  = 'outputs/'
SEED        = 0 if len(sys.argv) == 1 else int(sys.argv[1])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

################################## Load regression data
train_data = CopyMemory(MEM_LENGTH)
X_train_input, Y_train_input, X_train_target, Y_train_target = \
        train_data.sample_inputs_targets(NUM_TRAIN,normalise_x=True)

test_data = CopyMemory(MEM_LENGTH)
X_test_input, Y_test_input, X_test_target, Y_test_target = \
        test_data.sample_inputs_targets(NUM_TEST,normalise_x=True)

plot_data = CopyMemory(MEM_LENGTH)
X_plot_input, Y_plot_input, X_plot_target, Y_plot_target = \
        plot_data.sample_inputs_targets(NUM_PLOT,normalise_x=True)

markers = ['-', '--']
activations = [torch.nn.Sigmoid(), torch.nn.Sigmoid()]
error_fig = plt.figure()
# train/test error curves for standard and our init
experiment_data = np.zeros((4, MAX_EPOCHS)) 

for m_idx, initialise_as_glm in enumerate([True, False]):
    ################################## Initialise the Model
    if initialise_as_glm:
        x_init = [X_train_input, X_train_target]
        save_dir = 'glm_init/'
    else:
        x_init = None
        save_dir = 'not_glm_init/'

    cos_ker = lambda x1, x2: x1 @ x2.T / \
            (np.linalg.norm(x1, axis=1)*np.linalg.norm(x2.T, axis=0))
    k_delta = lambda x1, x2: (cos_ker(x1, x2) == 1).astype(np.float32)
    #k_delta = None
    f = FullyConnectedLayer(MEM_LENGTH+20, MEM_LENGTH+20, MEM_LENGTH+20, 
            x_init = x_init, kernel = k_delta, activation=activations[m_idx])

    model = DEQGLM(f, solver=None, tol=1e-2, max_iter=25, m=5)

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
            if not (lr_scheduler is None):
                lr_scheduler.step()
        
        total_loss += loss.item() * list(X.shape)[0]

        return total_loss / list(X.shape)[0]

    ################################## Optimise, minimising the loss
    opt = optim.Adam(model.parameters(), lr=1e-3)
    print("# Parmeters: ", sum(a.numel() for a in model.parameters()))


    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, MAX_EPOCHS*NUM_TRAIN, eta_min=1e-6)
    scheduler = None
    train_err = np.zeros((MAX_EPOCHS,))
    test_err = np.zeros((MAX_EPOCHS,))
    for i in range(MAX_EPOCHS):
        print(i)
        train_err[i] = epoch([Y_train_input, Y_train_target], model, opt, scheduler)
        test_err[i] = epoch([Y_test_input, Y_test_target], model)
    
    if PLOT:
        matplotlib_config()


        plot_1d_sequence_data(X_train_input, Y_train_input, 
                X_plot_input, Y_plot_input, save_dir + 'plot_input_train.pdf')
        plot_1d_sequence_data(X_train_target, Y_train_target, 
                X_plot_target, Y_plot_target, save_dir + 'plot_target_train.pdf')
        pred = model(Y_train_input).detach().numpy()
        plot_1d_sequence_data(X_train_target, pred, 
                X_plot_target, Y_plot_target, save_dir + 'plot_target_train_pred.pdf')
        plot_1d_sequence_data(X_test_input, Y_test_input, 
                X_plot_input, Y_plot_input, save_dir + 'plot_input_test.pdf')
        plot_1d_sequence_data(X_test_target, Y_test_target, 
                X_plot_target, Y_plot_target, save_dir + 'plot_target_test.pdf')
        pred = model(Y_test_input).detach().numpy()
        plot_1d_sequence_data(X_test_target, pred, 
                X_plot_target, Y_plot_target, save_dir + 'plot_target_test_pred.pdf')

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
