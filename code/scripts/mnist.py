import numpy as np
import sklearn as sklearn
from klr.klr import Klr
from klr.helpers import SquaredExponential
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from ..lib.deq import (FullyConnectedLayer, DEQGLM)
from ..lib.plotters import matplotlib_config

LAMB_SCALE = 2
MAX_EPOCHS = 5
BATCH_SIZE = 100
NUM_STEPS = 600

np.random.seed(0)
torch.manual_seed(0)

##################LOAD THE DATA
# Transform to normalized Tensors
transform = transforms.Compose([transforms.ToTensor()])
                                #transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('./MNIST/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST('./MNIST/', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
        shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=int(BATCH_SIZE/6),
        shuffle=False)

train_dataset_ = datasets.MNIST('./MNIST/', train=True, transform=transform, download=True)
train_loader_ = DataLoader(train_dataset, batch_size=60000,
        shuffle=False)
train_dataset_array = np.where(next(iter(train_loader_))[0].numpy() > 0, 1, 0)

Yhat = train_dataset_array.reshape((-1, 28*28)).astype(np.float32)
all_idx = list(range(0, 28))
Xs = np.transpose([np.tile(all_idx, len(all_idx)), np.repeat(all_idx,
    len(all_idx))]).astype(np.float32)


#################################### INITIALISE THE KLR MODEL
kernel = lambda x1, x2: (SquaredExponential(1)(x1,x2)
        + 0*np.eye(x1.shape[0])*10**(-8)).astype(np.float32)
K = kernel(Xs, Xs)
log_reg = Klr(precomputed_kernel=True)

lamb = np.amax(np.abs(np.linalg.eigvalsh(K)))


################################### INITIALISE THE DEQ MODEL
f = FullyConnectedLayer(784, 784, 784, activation = lambda z:
        (1+torch.exp(-z))**(-1), x_init
        = [torch.from_numpy(Xs.T),torch.from_numpy(Xs.T)], kernel=kernel,
        lamb_scale=LAMB_SCALE)
kern_model = DEQGLM(f, solver=None, tol=1e-20, max_iter=200, m=10)

g = FullyConnectedLayer(784, 784, 784, activation = lambda z:
        (1+torch.exp(-z))**(-1), x_init=False, kernel=None,
        lamb_scale=LAMB_SCALE)
rand_model = DEQGLM(g, solver=None, tol=1e-20, max_iter=200, m=10)


#################################### LOOP THROUGH THE DATA AND EVALUATE INITIAL
s_list = list(range(0,60000))
s_list_keep = [3678,13774,18222,19554,19857,22298,23396,38328,43401,48119,58961]
sum_err_kern = 0
sum_err_rand = 0

sum_err_kern_bin = 0
sum_err_rand_bin = 0
for s in s_list:
    Yhats = Yhat[s:s+1,:].T
    Ys = Yhats.copy()
    Ys[np.random.permutation(range(len(Ys)))[:int(len(Ys)/3)]]= \
        Ys[np.random.permutation(range(len(Ys)))[:int(len(Ys)/3)]] == 0
    if not (s in s_list_keep):
        continue

    log_reg.fit(K, Ys, num_iters=5, lamb=lamb*LAMB_SCALE)
    
    pred_klr = log_reg.predict_proba(K) 
    binary_pred_klr = np.where(pred_klr >= 0.5, 1, 0).reshape((28,28))

    pred_deq = (1+torch.exp(-kern_model(torch.from_numpy(Ys.T))))**(-1)
    binary_pred_kern = np.where(pred_deq[0] >= 0.5, 1, 0).reshape((28,28))

    pred_random = (1+torch.exp(-rand_model(torch.from_numpy(Ys.T))))**(-1)
    binary_pred_rand = np.where(pred_random[0] >= 0.5, 1, 0).reshape((28,28))
    
    err_kern = np.mean((pred_klr-pred_deq.detach().numpy())**2)
    err_rand = np.mean((pred_klr-pred_random.detach().numpy())**2)

    sum_err_kern = sum_err_kern + err_kern
    sum_err_rand = sum_err_rand + err_rand

    bin_kern = (np.sum(np.abs(binary_pred_kern - binary_pred_klr)))
    bin_rand = (np.sum(np.abs(binary_pred_rand - binary_pred_klr)))

    sum_err_kern_bin += bin_kern
    sum_err_rand_bin += bin_rand
    
    print(s)
    if bin_kern > 0:
        print('NOT THE SAME!')

        disagree_idx = np.nonzero(binary_pred_kern - binary_pred_klr)
        print((pred_deq.detach().numpy().reshape((28,28))[disagree_idx][0]).astype('str'))
        print((pred_klr).reshape((28,28))[disagree_idx][0].astype('str'))

        plt.imshow(binary_pred_klr, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('klr' + str(s) + '.pdf', bbox_inches='tight')
        plt.close()

        plt.imshow(binary_pred_kern, cmap='gray')
        plt.scatter(disagree_idx[1],disagree_idx[0], c='r')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('deq' + str(s) + '.pdf', bbox_inches='tight')
        plt.close()

        plt.imshow(binary_pred_rand, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('rand' + str(s) + '.pdf', bbox_inches='tight')
        plt.close()

print(sum_err_kern / len(s_list))
print(sum_err_rand / len(s_list))

print(sum_err_kern_bin / len(s_list))
print(sum_err_rand_bin / len(s_list))

quit()

################################## One training or testing iteration
def epoch(loader, loader_test, model1, model2, opt1=None, opt2=None):
    total_loss1 = np.zeros((2, int(60000/BATCH_SIZE)))
    total_loss2 = np.zeros((2, int(60000/BATCH_SIZE)))

    batch_num = 0
    for train, test in zip(loader, loader_test):
        Xtrain,ytrain = train; Xtest, ytest = test
        #model1.eval()
        #model2.eval()
        
        i = 0
        for X,Y in [(Xtest,ytest), (Xtrain,ytrain)]:
            bs = BATCH_SIZE if i == 1 else int(BATCH_SIZE/6)
            X = (X > 0.).to(X.dtype).reshape((bs, -1))

            X_noise = X.clone()

            X_noise[:,np.random.permutation(range(len(X[0])))[:int(len(X[0])/3)]]= \
                    (X[:,np.random.permutation(range(len(X[0])))[:int(len(X[0])/3)]] ==
                        0).to(X.dtype)

            #print(X[0].reshape((28,28)).numpy().astype(int))
            #print(X_noise[0].reshape((28,28)).numpy().astype(int))

            Xp = (1+torch.exp(-model1(X_noise)))**(-1)
            #print((Xp[0].reshape((28,28)).detach().numpy()>0.5).astype(int))
            if i == 0:
                Xp = Xp > 0.5
            loss = nn.MSELoss()(Xp,X)

            if i == 1:
                opt1.zero_grad()
                loss.backward()
                opt1.step()

            total_loss1[i,batch_num] = loss.item()

            Xp = (1+torch.exp(-model2(X_noise)))**(-1)
            #print((Xp[0].reshape((28,28)).detach().numpy()>0.5).astype(int))
            if i == 0:
                Xp = Xp > 0.5
            loss = nn.MSELoss()(Xp,X)
            if i == 1:
                opt2.zero_grad()
                loss.backward()
                opt2.step()

            
            total_loss2[i,batch_num] = loss.item()

            i += 1
        #print(total_loss1[:,:batch_num])
        #print(total_loss2[:,:batch_num])
        print(batch_num)
        batch_num = batch_num + 1

        if batch_num > NUM_STEPS:
            break
    
    return total_loss1, total_loss2

opt1 = optim.Adam(kern_model.parameters(), lr=1e-3)
opt2 = optim.Adam(rand_model.parameters(), lr=1e-3)

matplotlib_config()

for i in range(MAX_EPOCHS):
    kern_loss, rand_loss = epoch(train_loader, test_loader, kern_model, rand_model, opt1, opt2)
    np.save('kernel_loss' + str(i) + '.npy', kern_loss)
    np.save('rand_loss' + str(i) + '.npy', rand_loss)

    plt.plot(kern_loss[0,:NUM_STEPS], 'g')
    plt.plot(rand_loss[0,:NUM_STEPS], 'b')
    plt.xlabel('Training iteration')
    plt.ylabel('Test Error')
    plt.legend(['Kernel initialisation', 'Random Initialisation'], fontsize=20)
    plt.gca().set_yscale('log')
    plt.tight_layout()
    plt.savefig('test.pdf')
    plt.close()

    plt.plot(kern_loss[1,:NUM_STEPS], 'g')
    plt.plot(rand_loss[1,:NUM_STEPS], 'b')
    plt.xlabel('Training iteration')
    plt.ylabel('Train Error')
    plt.legend(['Kernel initialisation', 'Random Initialisation'], fontsize=20)
    plt.gca().set_yscale('log')
    plt.tight_layout()
    plt.savefig('train.pdf')
    plt.close()

