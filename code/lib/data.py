import numpy as np
import scipy.spatial as sp
import torch


# TODO make SyntheticData abstract class
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
            normalise_x = False, normalise_y = False, as_torch=True):
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

        if normalise_x:
            mean = np.mean(X_input, axis=1).reshape((-1,1))
            std = np.std(X_input, axis=1).reshape((-1,1))
            X_input = (X_input - mean)/std
            X_target = (X_target - mean)/std
        
        if as_torch:
            f = lambda x: torch.from_numpy(x.astype(np.float32))
        else:
            f = lambda x: x

        return [f(X_input), f(Y_input), f(X_target), f(Y_target)]

class CopyMemory(object):
    def __init__(self, T):
        self.T = T

    def sample_inputs_targets(self, num_samples, normalise_x = False,
            normalise_y = False, as_torch=True):
        assert normalise_y == False

        y_input = np.zeros((num_samples, self.T+20))
        y_input[:, 0:10] = np.random.randint(1, 9, size=(num_samples, 10))
        y_input[:, self.T+10] = 9
        
        y_target = np.zeros((num_samples, self.T+20))
        y_target[:, self.T+10:] = y_input[:,0:10]

        x_input = np.arange(0, self.T+20, 1).reshape((1, -1))
        x_input = np.tile(x_input, [num_samples, 1])
        x_target = np.arange(-self.T-10, 10, 1).reshape((1, -1))
        x_target = np.tile(x_target, [num_samples, 1])

        if normalise_x:
            mean = np.mean(x_target, axis=1).reshape((-1,1))
            std = np.std(x_target, axis=1).reshape((-1,1))
            x_input = (x_input - mean)/std
            x_target = (x_target - mean)/std

        if as_torch:
            f = lambda x: torch.from_numpy(x.astype(np.float32))
        else:
            f = lambda x: x

        return [f(x_input), f(y_input), f(x_target), f(y_target)]


        

