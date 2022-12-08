# import packages
import numpy as np

import skimage.io as io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import matplotlib.pyplot as plt
from  mpl_toolkits.axes_grid1 import ImageGrid

from admm_tv import *
from amatrix import *
from load_spectral import *

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * np.log((max_pixel ** 2) / (np.mean(np.square(y_pred - y_true))))) / 2.303

test_set = np.load("test_set.npy")

# Load images from hyperspectral dataset

test_set = [505, 870, 1178]

names, images = load_images(names=test_set)
# names, images = load_images()
print(images.shape)

# Number of channels
N = images.shape[3]

# noise parameter - standard deviation
sigma = 0.000

plot_on = True

extra=[0]

for e in extra:
    psnr = list()
    ssim = list()

    cpsnr = list()

    for i, img in enumerate(images):
        print(f"Processing image {i+1}/{len(images)}:")

        # Simulated measurements
        # e = int(img.shape[1]/(N)) # Extra rows to keep
        sim = SimulateImage(img, N, sigma, e)

        # sim = np.sum(sim, axis=2)

        # Plot each of the sampled channels
        # for i in range(sim.shape[2]):
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.imshow(sim[:,:,i], cmap="gray", vmin=0.0, vmax=1.0)
            # plt.show()

        # number of iterations for HQS/ADMM
        num_iters = 50

        # ADMM+TV solver
        rho = 16    # rho parameter of ADMM
        lam = 0.1   # relative weight of TV term

        x = list()
        for c in range(N):
            b = sim[:,:,c]
            # b = sim
            x.append(admm_tv(b, Afun, Atfun, lam, rho, num_iters, b.shape, c, N, e))

        x_admm_tv = np.stack(x, axis=2)

        # clip results to make sure it's within the range [0,1]
        x_admm_tv = np.clip(x_admm_tv, 0.0, 1.0)
        # compute PSNR using skimage library and round it to 2 digits
        PSNR_ADMM_TV = round(peak_signal_noise_ratio(img, x_admm_tv), 4)
        SSIM_ADAM_TV = round(structural_similarity(img, x_admm_tv), 4)

        print(f"psnr: {PSNR_ADMM_TV} \t ssim: {SSIM_ADAM_TV}")

        psnr.append(PSNR_ADMM_TV)
        ssim.append(SSIM_ADAM_TV)

        psnr2 = round(PSNR(img[np.newaxis,:,:,:], x_admm_tv[np.newaxis,:,:,:]), 4)
        # print(psnr2)
        cpsnr.append(psnr2)

        if (plot_on):
            fig = plt.figure(figsize=(20,6))
            grid = ImageGrid(fig, 111, nrows_ncols=(2, 10), axes_pad=0.1)

            plot_imgs = [np.dstack([img, x_admm_tv])[:,:,i] for i in
                    range(N*2)]

            for j, (ax, img) in enumerate(zip(grid, plot_imgs)):
                ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
                # ax.imshow(img)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

                if i == 0:
                    t = 505
                elif i == 1:
                    t = 870
                else:
                    t = 1178

                if j < N:
                    ax.set_title(f"Channel {i+1}")
                    io.imsave(f"images/{t}_channel{j}_source.png", img)
                else:
                    io.imsave(f"images/{t}_channel{j-10}_recon.png", img)


                if j==0:
                    ax.set_ylabel("Source")
                    ax.yaxis.set_visible(True)
                    ax.set_yticks([])
                    ax.set_yticks([],minor=True)
                if j==10:
                    ax.set_ylabel("Reconstructed")
                    ax.yaxis.set_visible(True)
                    ax.set_yticks([])
                    ax.set_yticks([],minor=True)


            # fig.tight_layout(h_pad=50)
            plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=1.5, wspace=1.5)

            plt.suptitle(f"ADMM+TV, PSNR= {PSNR_ADMM_TV}")
            plt.show()
            # plt.savefig("results/image_{i}.png", bbox_inches='tight')


    print("PSNRS:")
    print(psnr)
    print("SSIM:")
    print(ssim)
    print("CPSNRS:")
    print(cpsnr)

    res = np.stack([names, psnr, ssim, cpsnr])
    print(res)
    # np.save(f"results/admm_results_{e}.npy", res)

