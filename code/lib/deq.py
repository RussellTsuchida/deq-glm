import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import scipy.spatial as sp

class ResNetLayer(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, 
            num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, 
                padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, 
                padding=kernel_size//2, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, n_inner_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_channels)
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x):
        y = self.norm1(F.relu(self.conv1(z)))
        return self.norm3(F.relu(z + self.norm2(x + self.conv2(y))))

class FullyConnectedLayer(nn.Module):
    def __init__(self, num_in, width, num_out, activation=None, x_init=None):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.width  = width
        self._init_activation(activation)
        self._init_layers(x_init)

    def _init_layers(self, x_init):
        self.linear1 = nn.Linear(self.num_in, self.width, bias=False)
        self.linear2 = nn.Linear(self.num_in, self.width)
        self.linear3 = nn.Linear(self.width, self.num_out)

        if not (x_init is None):
            x_init = x_init - torch.mean(x_init)
            x_init = x_init/(torch.std(x_init))
            kernel = lambda x1, x2: np.exp(-\
                    sp.distance.cdist(x1, x2, 'sqeuclidean')/2)
            K = kernel(x_init.T, x_init.T)
            neg_K = lambda K: -torch.from_numpy(K.\
                    astype(np.float32))
            neg_K_norm = lambda K:  -torch.from_numpy(K.\
                    astype(np.float32))/self.width

            self.linear1.weight = nn.parameter.Parameter(neg_K_norm(K))
            self.linear2.weight = nn.parameter.Parameter(neg_K_norm(K))
            self.linear3.weight = nn.parameter.Parameter(neg_K_norm(K))

    def _init_activation(self, activation):
        if activation is None:
            activation = torch.nn.Tanh()
        self.activation = activation
        
    def forward(self, z, x):
        y = self.activation(self.linear1(z) + self.linear2(x))
        return self.linear3(y)

class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver=None, **kwargs):
        super().__init__()
        self.f = f
        self._init_solver(solver)
        self.kwargs = kwargs

    @staticmethod
    def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
        """ Anderson acceleration for fixed point iteration. """
        #bsz, d, H, W = x0.shape
        bsz = x0.shape[0]
        dimension = np.prod(list(x0[0,:].size()))

        X = torch.zeros(bsz, m, dimension, dtype=x0.dtype, device=x0.device)
        F = torch.zeros(bsz, m, dimension, dtype=x0.dtype, device=x0.device)
        X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
        X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
        
        H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
        H[:,0,1:] = H[:,1:,0] = 1
        y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
        y[:,0] = 1
        
        res = []
        for k in range(2, max_iter):
            n = min(k, m)
            G = F[:,:n]-X[:,:n]
            H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + \
                    lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
            alpha = torch.linalg.solve(H[:,:n+1,:n+1],y[:,:n+1])[:, 1:n+1, 0]

            X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + \
                    (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
            F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
            res.append((F[:,k%m] - X[:,k%m]).norm().item()/\
                    (1e-5 + F[:,k%m].norm().item()))
            if (res[-1] < tol):
                break
        return X[:,k%m].view_as(x0), res

    def _init_solver(self, solver):
        if solver is None:
            solver = self.anderson
        self.solver = solver

    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(\
                    lambda z: self.f(z, x),torch.zeros_like(x),**self.kwargs)
        z = self.f(z,x)
        
        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,x)
        def backward_hook(grad):
            g, self.backward_res = self.solver(\
                    lambda y: autograd.grad(f0, z0, y, 
                        retain_graph=True)[0] + grad, grad, **self.kwargs)
            return g
                
        z.register_hook(backward_hook)
        return z

