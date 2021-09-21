import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import copy
import scipy.spatial as sp
import scipy.special as spec

class ConvNet(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, 
                padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, 
                padding=kernel_size//2, bias=False)
        self.conv2.weight.data.normal_(0, 0.01)
        self.conv1.weight.data.normal_(0, 0.01)
        
    def forward(self, z, x):
        y = F.relu(self.conv1(z))
        return F.relu(z + self.conv2(y))

class DEQGLMConv(nn.Module):
    def __init__(self, num_in_channels, filter_size=3, solver=None, init_type='random',
        input_dim=(3,32,32), init_scale = 0.1, **kwargs):
        super().__init__()

        class ConvNet(nn.Module):
            def __init__(self, n_channels, kernel_size=3, act=None, init_type='random',
                init_scale = 0.01,input_dim=(3,32,32)):
                super().__init__()
                self.kernel_size = kernel_size
                self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, 
                        padding=kernel_size//2, bias=False)
                self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, 
                        padding=kernel_size//2, bias=False)

                self.init_params(init_type, input_dim, init_scale)
                self._init_act(act)

            def init_params(self, init_type, input_dim, init_scale):
                self._init_kernel()
                if init_type == 'random':
                    if not (init_scale is None):
                        self.conv1.weight.data.normal_(0, init_scale)
                        self.conv2.weight.data.normal_(0, init_scale)

                elif init_type == 'informed':
                    k1, l1  = self._kernel_and_spec_norm(self.kernel_size, self.conv1.weight,
                        input_dim)

                    lamb = l1/init_scale
                    neg_K_norm = lambda K_in: torch.from_numpy(-copy.deepcopy(K_in)/(lamb))
                    K_norm = lambda K_in:  torch.from_numpy(copy.deepcopy(K_in)/(lamb))

                    self.conv1.weight = nn.parameter.Parameter(neg_K_norm(k1))
                    self.conv2.weight = nn.parameter.Parameter(K_norm(k1))
                
                print(self._spec_norm(self.conv1.weight.detach().cpu().numpy(), input_dim))
                self.spec_norm = (\
                    self._spec_norm(self.conv1.weight.detach().cpu().numpy(), input_dim)+\
                    self._spec_norm(self.conv2.weight.detach().cpu().numpy(),input_dim))/2

            def _init_kernel(self):
                scaled_dist = lambda x1, x2, ls, nu: np.sqrt(2*nu)*\
                    (sp.distance.cdist(x1, x2, 'euclidean')/ls).astype(np.float32)

                self.matern = lambda x1, x2, var, ls, nu: var*\
                    spec.kv(nu, scaled_dist(x1, x2, ls, nu))*\
                    scaled_dist(x1, x2, ls, nu)**nu*\
                    (2**(1-nu))/(spec.gamma(nu))

            def _kernel_and_spec_norm(self, kernel_size, param, input_dim):
                assert not (input_dim is None)
                x = np.arange(0, kernel_size, 1).astype(np.int)
                X = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
                mid = (kernel_size + 1)/2-1
                mid = np.asarray([[mid, mid]])

                beta = 4
                alpha = 3
                ls_list = 1/np.random.gamma(alpha, 1/beta, param.shape[0])
                nu_list = np.random.exponential(0.5, param.shape[1])

                k = np.zeros((param.shape[0], param.shape[1], kernel_size, kernel_size),
                    dtype=np.float32)
                for ls_idx, ls in enumerate(ls_list):
                    for nu_idx, nu in enumerate(nu_list[:ls_idx+1]):
                        var = 1/np.random.gamma(alpha, 1/(beta))
                        ksub = self.matern(mid, X, var, ls, nu).\
                        reshape((kernel_size, kernel_size)).astype(np.float32)
                        ksub[int(mid[0,0]), int(mid[0,0])] = var
                        k[ls_idx, nu_idx, :, :] = ksub
                        k[nu_idx, ls_idx, :, :] = ksub
                
                return k, self._spec_norm(k, input_dim)

            def _spec_norm(self, k, input_dim):
                k_reshape = k.swapaxes(0,2)
                k_reshape = k_reshape.swapaxes(1,3)
                transforms = np.fft.fft2(k_reshape, input_dim, axes=[0, 1])
                svs = np.linalg.svd(transforms, compute_uv=False)
                lamb = np.amax(svs)
                return  lamb

            def _init_act(self, act):
                if act is None:
                    act = lambda x: x
                self.act = act
                
            def forward(self, z, y):
                return self.act(self.conv1(z) + self.conv2(y))

        
        self.act = F.relu
        self.conv_features = ConvNet(num_in_channels, filter_size, self.act, init_type=init_type,
            input_dim = input_dim, init_scale=init_scale)
        self.conv_output = ConvNet(num_in_channels, filter_size, init_type=init_type, 
            input_dim = input_dim, init_scale=init_scale)
        if init_type == 'informed':
            self.conv_output.conv1.weight = self.conv_features.conv1.weight
            self.conv_output.conv2.weight = self.conv_features.conv2.weight

        self.spec_norm = (self.conv_features.spec_norm + self.conv_output.spec_norm)/2
        self.deq = DEQFixedPoint(self.conv_features, solver, **kwargs)

    def forward(self, y):
        z = self.deq(y)
        return self.act(self.conv_output.conv1(z) + self.conv_output.conv2(y))

    def init_params(self, init_type, input_dim, init_scale, seed = None):
        if not (seed is None):
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.conv_features.init_params(init_type, input_dim, init_scale)
        self.conv_output.init_params(init_type, input_dim, init_scale)
        if init_type == 'informed':
            self.conv_output.conv1.weight = self.conv_features.conv1.weight
            self.conv_output.conv2.weight = self.conv_features.conv2.weight

        self.spec_norm = (self.conv_features.spec_norm + self.conv_output.spec_norm)/2


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

class ResNetLayerModified(nn.Module):
    def __init__(self, n_channels, n_inner_channels, kernel_size=3, 
            num_groups=8, init_as = 'random', input_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, 
                padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, 
                padding=kernel_size//2, bias=False)

        """
        self.conv3 = nn.Conv2d(n_inner_channels, n_channels, kernel_size, 
                padding=kernel_size//2, bias=False)
        self.conv4 = nn.Conv2d(n_channels, n_inner_channels, kernel_size, 
                padding=kernel_size//2, bias=False)

        self.norm1 = nn.GroupNorm(num_groups, n_channels)
        self.norm2 = nn.GroupNorm(num_groups, n_channels)
        self.norm3 = nn.GroupNorm(num_groups, n_inner_channels)
        """
        self._init_kernel(None)
        self._init_params(init_as, input_dim)

    def forward(self, z, y):
        #act = F.sigmoid
        act = F.leaky_relu
        #act = F.relu
        #act = lambda x: x
        """
        term3 = self.conv3(self.norm3(act(self.conv4(z))))
        term2 = self.norm2(self.conv2(y)+term3)
        term1 = self.conv1(z)

        return self.norm1(act(term1+term2))
        """
        return act(self.conv1(z)+self.conv2(y))

    def _init_kernel(self, kernel):
        if kernel is None:
            kernel = lambda x1, x2: (np.exp(-\
                    sp.distance.cdist(x1, x2, 'euclidean')/2)).\
                    astype(np.float32) 

            scaled_dist = lambda x1, x2, ls, nu: np.sqrt(2*nu)*\
                (sp.distance.cdist(x1, x2, 'euclidean')/ls).astype(np.float32)

            self.matern = lambda x1, x2, var, ls, nu: var*\
                spec.kv(nu, scaled_dist(x1, x2, ls, nu))*\
                scaled_dist(x1, x2, ls, nu)**nu*\
                (2**(1-nu))/(spec.gamma(nu))
                
        self.kernel = kernel

    def _init_params(self, init_as, input_dim):
        if init_as == 'random':
            #self.conv1.weight.data.normal_(0, 0.01)
            #self.conv2.weight.data.normal_(0, 0.01)
            #self.conv3.weight.data.normal_(0, 0.01)
            #self.conv4.weight.data.normal_(0, 0.01)
            pass
        elif init_as == 'informed':
            kernel_size = self.conv1.weight.shape[2]
            assert kernel_size == self.conv1.weight.shape[3]
            assert (kernel_size % 2)

            k1, l1 = self._kernel_and_spec_norm(kernel_size,
                self.conv1.weight,input_dim)
            k2, l2 = self._kernel_and_spec_norm(kernel_size,
                self.conv2.weight,input_dim)
            """
            k3, l3 = self._kernel_and_spec_norm(kernel_size,
                self.conv3.weight,input_dim)
            lamb = max([l1, l2, l3])
            """
            lamb = max([l1, l2])/50

            neg_K_norm = lambda K_in: torch.from_numpy(-copy.deepcopy(K_in)/(lamb))
            K_norm = lambda K_in:  torch.from_numpy(copy.deepcopy(K_in)/(lamb))

            self.conv1.weight = nn.parameter.Parameter(neg_K_norm(k1))
            self.conv2.weight = nn.parameter.Parameter(K_norm(k2))
            #self.conv3.weight = nn.parameter.Parameter(neg_K_norm(k3))
            
            """
            mid = int((kernel_size + 1)/2-1)
            c4 = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            c4[mid,mid] = -1
            c4 = np.tile(c4, 
                [self.conv4.weight.shape[0],
                            self.conv4.weight.shape[1], 1, 1])
            self.conv4.weight = nn.parameter.Parameter( torch.from_numpy(c4))
            """

    def _kernel_and_spec_norm(self, kernel_size, param, input_dim):
        assert not (input_dim is None)
        x = np.arange(0, kernel_size, 1).astype(np.int)
        X = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
        mid = (kernel_size + 1)/2-1
        mid = np.asarray([[mid, mid]])

        beta = 4
        alpha = 3
        ls_list = 1/np.random.gamma(alpha, 1/beta, param.shape[0])
        nu_list = np.random.exponential(0.5, param.shape[1])
        k = np.zeros((param.shape[0], param.shape[1], kernel_size, kernel_size),
            dtype=np.float32)
        for ls_idx, ls in enumerate(ls_list):
            for nu_idx, nu in enumerate(nu_list):
                var = 1/np.random.gamma(alpha, 1/(beta))
                ksub = self.matern(mid, X, var, ls, nu).\
                reshape((kernel_size, kernel_size)).astype(np.float32)
                ksub[int(mid[0,0]), int(mid[0,0])] = var
                k[ls_idx, nu_idx, :, :] = ksub

        transforms = np.fft.fft2(k, input_dim, axes=[0, 1])
        svs = np.linalg.svd(transforms, compute_uv=False)
        lamb = svs[0,0,0]
        return k, lamb

class FullyConnectedLayer(nn.Module):
    def __init__(self, num_in, width, num_out, activation=None, x_init=False, 
            kernel=None):
        """
        x_init (Bool or list(Torch Tensor)): If False, use random 
            initialisation. If True, use naive initialisation. If a list
            of pytorch tensors, use these to initialise the kernel matrix.
        """
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.width  = width

        self._init_kernel(kernel)
        self._init_activation(activation)
        self._init_layers(x_init)

    def _init_kernel(self, kernel):
        if kernel is None:
            kernel = lambda x1, x2: (np.exp(-\
                    sp.distance.cdist(x1, x2, 'sqeuclidean')/2)).\
                    astype(np.float32)
        self.kernel = kernel

    def _init_layers(self, x_init):
        self.x_init = x_init
        self.linear1 = nn.Linear(self.num_in, self.width, bias=False)
        self.linear2 = nn.Linear(self.num_in, self.width)
        self.linear3 = nn.Linear(self.width, self.num_out)
        self.linear4 = nn.Linear(self.num_in, self.num_out, bias = False)

        self.linear2.bias = nn.parameter.Parameter(torch.zeros((self.width)))
        self.linear3.bias = nn.parameter.Parameter(torch.zeros((self.num_out)))

        self.linear4.weight = \
            nn.parameter.Parameter(torch.zeros((self.num_in, self.num_out)))
        self.linear4.weight.requires_grad = False

        if not (x_init is False):
            if (x_init == True):
                x_in = torch.from_numpy(\
                        np.linspace(-2*np.pi, 2*np.pi, self.num_in).reshape((1,-1)))
                x_out = None
                x_init = [x_in, x_out]
            self._informed_init(x_init)

    def _informed_init(self, x_init):
        x = x_init[0]; x_star = x_init[1]

        K = self.kernel(x.numpy().T, x.numpy().T)
        lamb = (np.linalg.norm(K, ord=2))

        neg_K_norm = lambda K_in: torch.from_numpy(-copy.deepcopy(K_in)/(lamb))
        K_norm = lambda K_in:  torch.from_numpy(copy.deepcopy(K_in)/(lamb))
        
        self.linear1.weight = nn.parameter.Parameter(neg_K_norm(K))
        self.linear2.weight = nn.parameter.Parameter(K_norm(K))
        
        if not (x_star is None):
            K_star = self.kernel(x_star.numpy().T, x.numpy().T)
            self.linear3.weight = nn.parameter.Parameter(neg_K_norm(K_star))
            self.linear4.weight = nn.parameter.Parameter(K_norm(K_star))
        self.linear4.weight.requires_grad = True

    def _init_activation(self, activation):
        if activation is None:
            activation = lambda z: torch.nn.Tanh()(z)
        self.activation = activation
        
    def forward(self, z, x):
        y = self.activation(self.linear1(z) + self.linear2(x))
        return y

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

    @staticmethod
    def forward_iteration(f, x0, m=None, lam=None, max_iter=50, tol=1e-2, beta=None):
        f0 = f(x0)
        res = []
        for k in range(max_iter):
            x = f0
            f0 = f(x)
            res.append((f0 - x).norm().item() / (1e-5 + f0.norm().item()))
            if (res[-1] < tol):
                break
            if torch.isnan(f0).any() or torch.isinf(f0).any():
                break

        return f0, res

    def _init_solver(self, solver):
        if (solver is None) or (solver == 'naive'):
            solver = self.forward_iteration
        elif solver == 'anderson':
            solver = self.anderson
        self.solver = solver

    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        #x0 = torch.normal(0, 1, size=x.shape)
        x0 = torch.zeros_like(x)
        with torch.no_grad():
            z, err = self.solver(\
                    lambda z: self.f(z, x),x0,**self.kwargs)
        z = self.f(z,x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0,x)
        def backward_hook(grad):
            g, self.backward_res = self.solver(\
                    lambda y: autograd.grad(f0, z0, y, 
                        retain_graph=True)[0] + grad, grad, **self.kwargs)
            return g
        try:
            z.register_hook(backward_hook)
        except:
            print('warning')
        return z

class DEQGLM(nn.Module):
    def __init__(self, f, solver=None, **kwargs):
        super().__init__()
        self.deq = DEQFixedPoint(f, solver, **kwargs)

    def forward(self, x):
        z = self.deq(x)
        return self.deq.f.linear3(z) + self.deq.f.linear4(x)

