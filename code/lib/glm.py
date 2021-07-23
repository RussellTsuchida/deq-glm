import numpy as np
import scipy.spatial as sp
import sklearn.kernel_ridge as krr

class Glm(object):
    def __init__(self, a=None, adash = None, addash = None, T=None, lamb = 1, 
            kernel=None):
        self._init_funs(a, adash, addash, T)
        self._init_kernel(kernel)
        self.lamb = lamb

    def _init_funs(self, a, adash, addash, T):
        if a is None:
            assert (adash is None) and (addash is None)
            a = lambda z: np.log(2*np.cosh(z))
            adash = np.tanh
            addash = lambda z: (1/np.cosh(z))**2
        if T is None:
            T = lambda z: z

        self.a = a
        self.adash = adash
        self.addash = addash
        self.T = T

    def _init_kernel(self, kernel):
        if kernel is None:
            kernel = lambda x1, x2: 0.1*np.exp(-\
                    sp.distance.cdist(x1, x2, 'sqeuclidean'))
        self.kernel = kernel

    def fit(self, X, Y):
        K = self.kernel(X, X)
        #alpha = np.zeros((X.shape[0], 1))
        alpha = np.random.normal(0, 1, (X.shape[0], 1))/np.sqrt(X.shape[0])

        Ka = K @ alpha
        for c in range(10):
            D = np.diag(self.addash(Ka).reshape((-1,)))

            J = K @ D @ K + self.lamb*K
            z = K @ alpha + np.linalg.inv(D) @ (Y - self.adash(Ka))
            alpha = np.linalg.solve(J, K @ D @ z)

        self.alpha = alpha
        self.X = X

        return alpha

    def predict(self, Xstar):
        Kstar = self.kernel(Xstar, self.X)
        return Kstar @ self.alpha

    def fit_and_predict(self, X, Y, Xstar):
        self.fit(X, Y)
        return self.predict(Xstar)

    def __call__(self, Xstar):
        return self.predict(Xstar)

if __name__ == '__main__':
    from ..lib.data import SequenceOneDimension
    from ..lib.plotters import (matplotlib_gc, matplotlib_config,
            plot_1d_sequence_data)

    NUM_TRAIN   = 101
    NUM_TEST    = 1000
    NUM_PLOT    = 100
    DIMENSION_Y = 100
    NOISE_VAR   = 0.01
    TARGET_FUN  = lambda x: np.exp(-np.abs(x)) * np.sin(x) + np.exp(-(x+9)**2)
    OFFSET      = 2


    # Training data
    train_data = SequenceOneDimension(-2*np.pi, 2*np.pi, DIMENSION_Y, OFFSET,
        NOISE_VAR)
    X_train_input, Y_train_input, X_train_target, Y_train_target = \
            train_data.sample_inputs_targets(NUM_TRAIN, TARGET_FUN, as_torch=False)
    plot_data = SequenceOneDimension(-2*np.pi, 2*np.pi, DIMENSION_Y, OFFSET,
        0)
    X_plot_input, Y_plot_input, X_plot_target, Y_plot_target = \
            plot_data.sample_inputs_targets(NUM_PLOT, TARGET_FUN, as_torch=False)

    # Model
    """
    a = lambda z: np.log(1+np.exp(z))
    adash = lambda z: np.exp(z)/(1+np.exp(z))
    addash = lambda z: np.exp(z)/(1+np.exp(z))**2

    a = lambda z: z**2/2
    adash = lambda z: z
    addash = lambda z: np.ones_like(z)

    """
    a = None
    adash = None
    addash = None

    model = Glm(a = a, adash = adash, addash = addash)
    #model = krr.KernelRidge(kernel='rbf')
    model.fit(X_train_input[0,:].reshape((-1,1)), Y_train_input[0,:].reshape((-1,1)))
    pred = model.predict(X_train_target[0,:].reshape((-1,1)))

    # Plot
    matplotlib_config()
    save_dir = 'glm/'

    xlim = [-2*np.pi-OFFSET, 2*np.pi]
    ylim = [-5, 5]

    plot_1d_sequence_data(X_train_input, Y_train_input,
            X_plot_input, Y_plot_input, save_dir + 'plot_input_train.pdf', xlim, ylim)
    plot_1d_sequence_data(X_train_target, Y_train_target,
            X_plot_target, Y_plot_target, save_dir + 'plot_target_train.pdf', xlim, ylim)
    plot_1d_sequence_data(X_train_target[0,:].reshape((1,-1)), pred.reshape((1,-1)),
            X_plot_target, Y_plot_target, save_dir + 'plot_target_train_pred.pdf', xlim, ylim)

