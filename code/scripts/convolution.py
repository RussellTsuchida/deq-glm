import numpy as np
import scipy.spatial as sp
import matplotlib.pyplot as plt
import torchvision
import matplotlib.patches as patches

from ..lib.plotters import matplotlib_config

matplotlib_config()

w = 32
h = w
fsize = 5

x1 = np.arange(0, w, 1)
x2 = np.arange(0, h, 1)
X = np.transpose([np.tile(x1, len(x2)), np.repeat(x2, len(x1))])

norm = lambda x1, x2: sp.distance.cdist(x1, x2, metric='chebyshev')
sqexp = lambda x1, x2: sp.distance.cdist(x1, x2, metric='sqeuclidean')
k = lambda x1, x2: np.exp(-sqexp(x1,x2)/20)*( norm(x1,x2) <= fsize)
K = k(X, X)

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                            download=True)

# Filter 1
f1_centre = [10, 2]
f1_bot_left = [f1_centre[0] - fsize, f1_centre[1] - fsize]
f1_1d = f1_centre[1]*w + f1_centre[0]

# Filter 2
f2_centre = [20, 17]
f2_bot_left = [f2_centre[0] - fsize, f2_centre[1] - fsize]
f2_1d = f2_centre[1]*w + f2_centre[0]

fheight = fsize*2 + 1
fwidth = fheight

# Plot as a matrix-vector product
fig, ax = plt.subplots(1,2, sharey=True)

image = np.array(trainset[0][0])
image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
flat_image = np.reshape(image, (-1,1))

ax[1].imshow(flat_image, aspect=1/50)
ax[1].axis('off')
ax[1].annotate('$Y_i$', xy=(0.5, -0.05), xytext=(0.5, -0.2),
            fontsize=14, ha='center', va='bottom', xycoords='axes fraction',
            arrowprops=dict(arrowstyle='-[, widthB=1.0', lw=2.0))




rect = patches.Rectangle((0,f1_1d), w*h,10, linewidth=1, edgecolor='r', facecolor='none')
ax[0].add_patch(rect)

rect = patches.Rectangle((0,f2_1d), w*h,10, linewidth=1, edgecolor='y', facecolor='none')
ax[0].add_patch(rect)

ax[0].imshow(K, aspect=1)
ax[0].axis('off')
ax[0].annotate('$K(X_i,X_i)$', xy=(0.5, -0.05), xytext=(0.5, -0.2),
            fontsize=14, ha='center', va='bottom', xycoords='axes fraction',
            arrowprops=dict(arrowstyle='-[, widthB=8.0', lw=2.0))


plt.subplots_adjust(wspace=-0.41)
plt.savefig('conv_mat.pdf', bbox_inches='tight')
plt.close()


# Plot as a convolution
fig, ax = plt.subplots(1, 2)

rect = patches.Rectangle(f1_bot_left, fwidth, fheight, linewidth=3, edgecolor='r', facecolor='none')
ax[1].add_patch(rect)

rect = patches.Rectangle(f2_bot_left, fwidth, fheight, linewidth=3, edgecolor='y', facecolor='none')
ax[1].add_patch(rect)

ax[1].imshow(image, cmap='gray')
ax[1].axis('off')
plt.tight_layout()

ax[0].imshow(K[f1_1d,:].reshape((w,h))+K[f2_1d,:].reshape((w,h)))
ax[0].axis('off')
rect = patches.Rectangle(f1_bot_left, fwidth, fheight, linewidth=3, edgecolor='r', facecolor='none')
ax[0].add_patch(rect)

rect = patches.Rectangle(f2_bot_left, fwidth, fheight, linewidth=3, edgecolor='y', facecolor='none')
ax[0].add_patch(rect)

"""
ax[0].annotate('$W$', xy=(0.5, -0.05), xytext=(0.5, -0.2),
            fontsize=14, ha='center', va='bottom', xycoords='axes fraction',
            arrowprops=dict(arrowstyle='-[, widthB=7.0', lw=2.0))
"""
ax[0].annotate('$W$', xy=(0.5, 0.5), xytext=(0.35, 0.65), 
            fontsize=14, ha='center', va='bottom', xycoords='axes fraction',
            color='red')
ax[0].annotate('$W$', xy=(0.5, 0.5), xytext=(0.65, 0.18), 
            fontsize=14, ha='center', va='bottom', xycoords='axes fraction',
            color='yellow')

ax[1].annotate('$Y_i$', xy=(0.5, -0.05), xytext=(0.5, -0.2),
            fontsize=14, ha='center', va='bottom', xycoords='axes fraction',
            arrowprops=dict(arrowstyle='-[, widthB=7.0', lw=2.0))

plt.savefig('conv.pdf', bbox_inches='tight')


