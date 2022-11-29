###################################################
#   Release code for CSC2529 HW6, task 3
#
#   Instructions:
#       You don't need to change anything here, please
#       edit the files leastnorm.py, admm_tv.py, and
#       admm_dncnn.py
#
#   Gordon Wetzstein, 10/2021
#   David Lindell adapted for CSC2529 10/2022
###################################################

# import packages
import numpy as np

import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio

import matplotlib.pyplot as plt

from admm_tv import *
from amatrix import *

# select target image and load it
img = io.imread('birds.png').astype(float)/255.

# noise parameter - standard deviation
sigma = 0.001

# Simulated measurements
b = Afun(img) + sigma * np.random.randn(img.shape[0], img.shape[1])

# number of iterations for HQS/ADMM
num_iters = 10

# ADMM+TV solver
rho = 16    # rho parameter of ADMM
lam = 1.0   # relative weight of TV term
x_admm_tv = admm_tv(b, Afun, Atfun, lam, rho, num_iters, img.shape,
        anisotropic_tv=False)
# x_admm_tv = admm_tv(b, Afun, Atfun, lam, rho, num_iters, img.shape)
# clip results to make sure it's within the range [0,1]
x_admm_tv = np.clip(x_admm_tv, 0.0, 1.0)
# compute PSNR using skimage library and round it to 2 digits
PSNR_ADMM_TV = round(peak_signal_noise_ratio(img, x_admm_tv), 1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(x_admm_tv, cmap="gray", vmin=0.0, vmax=1.0)
ax.set_title("ADMM+TV, PSNR: " + str(PSNR_ADMM_TV))
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

plt.show()

