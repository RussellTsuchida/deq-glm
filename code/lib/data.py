import numpy as np
import scipy.spatial as sp
import torch
from torch.utils.data import Dataset
import tifffile
import os
import natsort

# TODO make SyntheticData abstract class
class SequenceOneDimension(object):
    def __init__(self, xmin, xmax, dimension_y, offset, noise_var):
        self.x_min = xmin
        self.x_max = xmax
        self.dimension_y = dimension_y
        self.offset = offset
        self.noise_var = noise_var
        self.K = None

    def _sample_random_gp(self, x, num_samples, kernel=None):
        if self.K is None:
            if kernel is None:
                kernel = lambda x1, x2: self.noise_var*\
                        np.exp(-sp.distance.cdist(x1, x2, 'sqeuclidean')/2)*\
                        np.exp(-np.sin(sp.distance.cdist(x1, x2, 'euclidean'))**2)
                """
                kernel = lambda x1, x2: self.noise_var*\
                        np.exp(-sp.distance.cdist(x1, x2, 'sqeuclidean')/2)
                """
            self.K = kernel(x, x)

        # Generate a (num_samples, x.shape[0]) np array of samples
        samples = np.random.multivariate_normal(np.zeros((x.shape[0],)), self.K,
                size = num_samples)

        return samples

    def sample_inputs_targets(self, num_samples, target_fun, 
            normalise_x = False, normalise_y = False, as_torch=True):
        X_input = np.tile(np.linspace(self.x_min, self.x_max, self.dimension_y),
                    [num_samples, 1])
        X_target = np.tile(np.linspace(self.x_min-self.offset, self.x_max-self.offset, 
            self.dimension_y), [num_samples, 1])

        x_both = np.hstack((X_input[0,:], X_target[0,:])).reshape((-1,1))

        gp_samples = self._sample_random_gp(x_both, num_samples)

        gp_input = gp_samples[:, :self.dimension_y]
        gp_target = gp_samples[:, self.dimension_y:]

        Y_input = target_fun(X_input) + gp_input
        Y_target = target_fun(X_target) + gp_target

        if normalise_y:
            if (normalise_y == True):
                mean = np.mean(Y_input)
                std = np.std(Y_input)
            else:
                mean = normalise_y[0]; std = normalise_y[1]
            Y_input = (Y_input - mean)/std
            Y_target = (Y_target - mean)/std

            self.mean_y = mean
            self.std_y = std

        if normalise_x:
            if (normalise_x == True):
                mean = np.mean(X_input)
                std = np.std(X_input)
            else:
                mean = normalise_x[0]; std = normalise_x[1]
            X_input = (X_input - mean)/std
            X_target = (X_target - mean)/std
            self.mean_x = mean
            self.std_x = std
        
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

        y_input = self._as_one_hot(y_input)
        y_target = self._as_one_hot(y_target)

        x_input = np.arange(0, self.T+20, 1).reshape((1, -1))
        x_input = np.tile(x_input, [num_samples, 1])
        x_target = np.arange(-self.T-10, 10, 1).reshape((1, -1))
        x_target = np.tile(x_target, [num_samples, 1])

        if normalise_x:
            mean = np.mean(x_target)
            std = np.std(x_target)
            x_input = (x_input - mean)/std
            x_target = (x_target - mean)/std

        if as_torch:
            f = lambda x: torch.from_numpy(x.astype(np.float32))
        else:
            f = lambda x: x

        return [f(x_input), f(y_input), f(x_target), f(y_target)]

    def _as_one_hot(self, x, num_classes=10):
        ret = np.eye(num_classes)[x.astype(np.int)]
        ret = np.swapaxes(ret, 1, 2)
        ret = np.swapaxes(ret, 0, 2)
        return ret
        
class HyperSpectralData(Dataset):
    def __init__(self, main_dir, transform, num_channels=None):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.num_channels = num_channels

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        tiffloader = lambda x: tifffile.imread(x)
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = (tiffloader(img_loc)/255).astype(np.float32)
        if not (self.num_channels is None):
            image = image[:self.num_channels, :, :]
        image = np.swapaxes(image, 0, 2)
        tensor_image = self.transform(image)
        # Return with a dummy class of 0 to make compatible with cifar
        return [tensor_image, 0]
