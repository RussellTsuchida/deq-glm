import numpy as np
import time
import scipy.spatial as sp
import scipy
import matplotlib.pyplot as plt
from ..lib.deq import DEQGLMConv
from ..lib.plotters import matplotlib_config

NUM_TRIALS = 100
MATRIX_SIZE_MAX = 1000
MATRIX_SIZE_MIN = 50
MATRIX_SIZE_STP = 50

NUM_CHANNELS_MAX = 64
NUM_CHANNELS_MIN = 1
NUM_CHANNELS_STP = 1
IMAGE_SIZE = 32

#MODE = 'MLP' 
MODE = 'CONV'

if MODE == 'CONV':
    times_rand = []
    times_kern = []
    matrix_sizes = list( range(NUM_CHANNELS_MIN, NUM_CHANNELS_MAX,
        NUM_CHANNELS_STP))
    for num_channels in matrix_sizes:
        print(num_channels)
        starttime = time.time()
        for i in range(NUM_TRIALS):
            model = DEQGLMConv(num_channels, 5, init_type='informed', 
                    input_dim=(IMAGE_SIZE, IMAGE_SIZE),
            init_scale = 1, num_hidden=None)
        endtime = time.time()
        times_kern.append( (endtime-starttime)/NUM_TRIALS)

        starttime = time.time()
        for i in range(NUM_TRIALS):
            model = DEQGLMConv(num_channels, 5, init_type='random', 
                    input_dim=(IMAGE_SIZE, IMAGE_SIZE),
            init_scale = 1, num_hidden=None)
        endtime = time.time()
        times_rand.append( (endtime-starttime)/NUM_TRIALS)


else:
    times_rand = []
    times_kern = []
    matrix_sizes = list( range(MATRIX_SIZE_MIN, MATRIX_SIZE_MAX, MATRIX_SIZE_STP))

    for MATRIX_SIZE in matrix_sizes:
        print(MATRIX_SIZE)

        starttime = time.time()
        for i in range(NUM_TRIALS):
            W = np.random.normal(0, 1, (MATRIX_SIZE,
                MATRIX_SIZE))/np.sqrt(MATRIX_SIZE)
            V = np.random.normal(0, 1, (MATRIX_SIZE,
                MATRIX_SIZE))/np.sqrt(MATRIX_SIZE)
            #W = np.random.uniform(-1, 1, (MATRIX_SIZE,
            #    MATRIX_SIZE))/np.sqrt(MATRIX_SIZE)
            #V = np.random.uniform(-1, 1, (MATRIX_SIZE,
            #    MATRIX_SIZE))/np.sqrt(MATRIX_SIZE)
        endtime = time.time()
        times_rand.append((endtime - starttime)/NUM_TRIALS)

        starttime = time.time()
        for i in range(NUM_TRIALS):
            X = np.linspace(0, MATRIX_SIZE).reshape((-1,1))
            Xtilde = np.linspace(MATRIX_SIZE, 2*MATRIX_SIZE).reshape((-1,1))
            K = np.exp(-sp.distance.cdist(X, X, 'sqeuclidean'))
            Ktilde = np.exp(-sp.distance.cdist(Xtilde, X, 'sqeuclidean'))

            spec_norm = max(np.abs(np.linalg.eigvalsh(K)))
            W = K / spec_norm
            V = Ktilde / spec_norm
        endtime = time.time()
        times_kern.append((endtime - starttime)/NUM_TRIALS)

matplotlib_config()
plt.plot(matrix_sizes, times_rand, 'b')
plt.plot(matrix_sizes, times_kern, 'g')
if MODE == 'CONV':
    plt.xlabel(r'\# Channels')
elif MODE == 'MLP':
    plt.xlabel(r'Width')
plt.ylabel(r'Time (seconds)')
plt.gca().set_yscale('log')
plt.tight_layout()
plt.legend(['Random initialisation', 'Kernel initialisation'], fontsize=20,
        loc='upper left')
plt.savefig('times' + MODE + '.pdf')
