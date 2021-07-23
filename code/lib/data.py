import numpy as np
import scipy.spatial as sp
import torch

class SequenceOneDimension(object):
    def __init__(self, xmin, xmax, dimension_y, offset, noise_var):
        self.xmin = xmin
        self.xmax = xmax
        self.dimension_y = dimension_y
        self.offset = offset
        self.noise_var = noise_var
        self.K = None

    def _sample_random_gp(self, x, num_samples, kernel=None):
        if self.K is None:
            if kernel is None:
                kernel = lambda x1, x2: self.noise_var*\
                        np.exp(-sp.distance.cdist(x1, x2, 'sqeuclidean')/2)
            self.K = kernel(x, x)

        # Generate a (num_samples, x.shape[0]) np array of samples
        samples = np.random.multivariate_normal(np.zeros((x.shape[0],)), self.K,
                size = num_samples)

        return samples

    def sample_inputs_targets(self, num_samples, target_fun, 
            normalise_x = True, normalise_y = True, as_torch=True):
        X_input = np.tile(np.linspace(-2*np.pi, 2*np.pi, self.dimension_y),
                    [num_samples, 1])
        X_target = np.tile(np.linspace(-2*np.pi-self.offset, 2*np.pi-self.offset, 
            self.dimension_y), [num_samples, 1])

        x_both = np.hstack((X_input[0,:], X_target[0,:])).reshape((-1,1))

        gp_samples = self._sample_random_gp(x_both, num_samples)

        gp_input = gp_samples[:, :self.dimension_y]
        gp_target = gp_samples[:, self.dimension_y:]

        Y_input = target_fun(X_input) + gp_input
        Y_target = target_fun(X_target) + gp_target

        if normalise_y:
            mean = np.mean(Y_input, axis=1).reshape((-1,1))
            std = np.std(Y_input, axis=1).reshape((-1,1))
            Y_input = (Y_input - mean)/std
            Y_target = (Y_target - mean)/std
        
        if as_torch:
            f = lambda x: torch.from_numpy(x.astype(np.float32))
        else:
            f = lambda x: x

        return [f(X_input), f(Y_input), f(X_target), f(Y_target)]
        

