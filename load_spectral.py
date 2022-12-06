import skimage.io as io
import matplotlib.pyplot as plt
import os
import numpy as np
import tiff

def load_images(path="/mnt/Data/Media/Images/data_cleaned/", dataset = "BGU-HS",
        num_images=3, names=list()):

    # Get all BGU files
    files = [f for f in os.listdir(path) if f.split("_")[-1] == f"{dataset}.tiff"]

    # Unique image names
    if len(names) == 0:
        names = np.unique([f.split("_")[0] for f in files])[:num_images]

    # Keep every 3rd channel
    channel_list = list(range(0, 30, 3))

    images = list()
    for name in names:
        img = [np.array(io.imread(path+ \
                f'{name}_{ch}_{dataset}.tiff')).astype("float64")[:480, :480]
                    for ch in channel_list]
        images.append(np.stack(img, axis=2))

    images = np.stack(images, axis=0)

    return names, images

if __name__=="__main__":

    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    _, images = load_images()

    i1 = images[0,:,:,8]
    i2 = images[0,:,:,9]

    p = psnr(i1, i2)
    s = ssim(i1, i2)

    print(p,s)

