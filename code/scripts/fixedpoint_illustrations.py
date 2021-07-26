import numpy as np
import matplotlib.pyplot as plt

from ..lib.plotters import matplotlib_config

NUM_POINTS  = 100
DIR = 'outputs/fixedpoint_illustrations/'

matplotlib_config()

y = np.linspace(-2, 2, NUM_POINTS)
lin = lambda theta, y: theta[0]*y + theta[1]
g = lambda theta, y: np.tanh(lin(theta, y))
relu = lambda theta, y: (lin(theta, y) > 0) * lin(theta, y)

def plot_fixedpoint(y, f, fname, ylabel):
    plt.plot(y, f)
    y = np.linspace(-1.1, 1.1, NUM_POINTS)
    plt.plot(y, y, ls='--')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.ylabel(ylabel)
    plt.xlabel(r'$y$')
    plt.xlim([-2, 2])
    plt.ylim([-1.1, 1.1])
    plt.tight_layout()
    plt.savefig(DIR+fname)
    plt.close()

# Case with unique fixed point
theta = [1, 0]
f = g(theta, y)
plot_fixedpoint(y, f, 'unique1.pdf', ylabel=r'$g(y)=\tanh(y)$')

# Case with unique fixed point
theta = [0.9, 1]
f = g(theta, y)
plot_fixedpoint(y, f, 'unique2.pdf', ylabel=r'$g(y)=\tanh(0.9y+2)$')

# Case with unique fixed point
theta = [3, 0]
f = -g(theta, y)
plot_fixedpoint(y, f, 'unique3.pdf', ylabel=r'$g(y)=\tanh(-3y)$')

# Case with more than one fixed point
theta = [3, 0]
f = g(theta, y)
plot_fixedpoint(y, f, 'not_unique.pdf', ylabel=r'$g(y)=\tanh(3y)$')

# Case with no fixed point
theta = [1, 0.1]
f = relu(theta, y)
plot_fixedpoint(y, f, 'none.pdf', ylabel=r'$g(y)=\text{ReLU}(y+0.1)$')

# Case with infinite number of fixed points
theta = [1, 0.]
f = relu(theta, y)
plot_fixedpoint(y, f, 'infinite.pdf', ylabel=r'$g(y)=\text{ReLU}(y)$')

