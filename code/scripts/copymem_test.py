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
MEM_LENGTH  = 50
MAX_EPOCHS  = 100
PLOT        = True
OUTPUT_DIR  = 'outputs/copymem/'
SEED        = 0 if len(sys.argv) == 1 else int(sys.argv[1])
FREEZE_FOR  = 20

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)
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
        plot_data.sample_inputs_targets(1,normalise_x=True)


markers = ['-', '--', '-.']
activations = [torch.nn.Sigmoid(), torch.nn.Sigmoid(), torch.nn.Sigmoid()]
#activations = [None, None, None]
error_fig = plt.figure()
# train/test error curves for standard and our init
experiment_data = np.zeros((6, MAX_EPOCHS)) 

for m_idx, initialise_as_glm in enumerate(['informed', 'naive', 'random']):
    ################################## Initialise the Model
    kernel = None
    if initialise_as_glm == 'informed':
        x_init = [X_train_input, X_train_target]
        save_dir = 'glm_init_informed/'
        cos_ker = lambda x1, x2: x1 @ x2.T / \
                (np.linalg.norm(x1, axis=1)*np.linalg.norm(x2.T, axis=0))
        kernel = lambda x1, x2: ( np.abs(cos_ker(x1, x2) - 1) <= 1e-3).astype(np.float32)
    elif initialise_as_glm == 'naive':
        x_init = True
        save_dir = 'glm_init_naive/'
    else:
        x_init = False
        save_dir = 'not_glm_init/'

    f = FullyConnectedLayer(MEM_LENGTH+20, MEM_LENGTH+20, (MEM_LENGTH+20)*10, 
            x_init = x_init, kernel = kernel, activation=activations[m_idx])

    model = DEQGLM(f, solver=None, tol=1e-2, max_iter=25, m=5)

    ################################## One training or testing iteration
    def epoch(data, model, opt=None):
        total_loss = 0.
        model.eval() if opt is None else model.train()

        X, y = data
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp,y)
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
        print(i, flush=True)
        if (initialise_as_glm == 'naive') and (i < FREEZE_FOR):
            for param in model.deq.f.linear1.parameters():
                param.requires_grad = False
            for param in model.deq.f.linear2.parameters():
                param.requires_grad = False
        else:
            for param in model.deq.f.linear1.parameters():
                param.requires_grad = True
            for param in model.deq.f.linear2.parameters():
                param.requires_grad = True
        
        train_err[i] = epoch([Y_train_input, Y_train_target], model, opt)
        test_err[i] = epoch([Y_test_input, Y_test_target], model)

	# Plot some sample trajcetories
        if PLOT and i in [0, FREEZE_FOR, MAX_EPOCHS-1]:
            xlim=[1.5, 2]
            plot_1d_sequence_data(X_train_input, Y_train_input, 
                    X_plot_input, Y_plot_input, 
                    save_dir + str(i) + 'plot_input_train.pdf', xlim=xlim)
            plot_1d_sequence_data(X_train_target, Y_train_target, 
                    X_plot_target, Y_plot_target, 
                    save_dir + str(i) + 'plot_target_train.pdf', xlim=xlim)
            pred = model(Y_train_input).detach().numpy()
            plot_1d_sequence_data(X_train_target, pred, 
                    X_plot_target, Y_plot_target, 
                    save_dir + str(i) + 'plot_target_train_pred.pdf', xlim=xlim)
            plot_1d_sequence_data(X_test_input, Y_test_input, 
                    X_plot_input, Y_plot_input, 
                    save_dir + str(i) + 'plot_input_test.pdf', xlim=xlim)
            plot_1d_sequence_data(X_test_target, Y_test_target, 
                    X_plot_target, Y_plot_target, 
                    save_dir + str(i) + 'plot_target_test.pdf', xlim=xlim)
            pred = model(Y_test_input).detach().numpy()
            plot_1d_sequence_data(X_test_target, pred, 
                    X_plot_target, Y_plot_target, 
                    save_dir + str(i) + 'plot_target_test_pred.pdf', xlim=xlim)
    
    if PLOT:
        matplotlib_config()
	
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
