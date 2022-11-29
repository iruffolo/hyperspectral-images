#!/usr/bin/python3

# import packages
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import cv2

def make_mask(img_size, start_row=0, extra_rows=0, stride=3):
    '''
    Returns mask the same size as img_size.

    Inputs:
        img_size: (height, width) of image to mask
        start_row: first row to be accepted (default 0)
        extra_rows: additional rows if mask wants to include consecutive rows
        stride: number of rows to skip before taking next row (exluding the
        extra_rows taken)

    Outputs:
        return: Image of size (height, width) with 1's in rows that are to be
        taken, and zeros everywhere else.
    '''

    mask = np.zeros((img_size[0], img_size[1]))

    for row in range(start_row, img_size[0], stride+extra_rows*stride):
        mask = cv2.rectangle(mask, (0,row), (img_size[1], row+extra_rows), 1, -1)

    return mask

def Atfun(x, channels=3):
    '''
    Calculates Atx where At is the transpose of the image formation matrix A,
    and x is an input image.

    Inputs:
    x: Image of size (height, width)
    channels: Number of channels in the original image

    Outputs:
    return: New image of size (height, width, channels)
    '''

    # Number of hyperspectral channels
    # channels = x.shape[2] if len(x.shape) > 2 else 1

    masks = [make_mask((x.shape[0],x.shape[1]), d, 0, channels)
                for d in range(channels)]

    # Stack masks into np array
    masks = np.stack(masks, axis=2)

    return masks * x[:,:,np.newaxis]


def Afun(x):
    '''
    Calculates Ax where A is the image formation matrix, and x is an input
    image.

    Inputs:
    x: Image of size (height, width, d) with d channels

    Outputs:
    return: New image of size (height, width) sampled from the d channels of x
    '''

    # Number of hyperspectral channels
    channels = x.shape[2] if len(x.shape) > 2 else 1

    masks = [make_mask((x.shape[0],x.shape[1]), d, 0, channels)
                for d in range(channels)]

    # Stack masks into np array
    masks = np.stack(masks, axis=2)

    # Mupltiply each image with its corresponding mask, then sum all masked
    # images into final image.
    return np.sum(masks * x, axis=2)


if __name__=="__main__":

    img = io.imread('birds.png').astype(float)/255.

    ares = Afun(img)
    atres = Atfun(ares)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(ares, cmap="gray", vmin=0.0, vmax=1.0)
    ax = fig.add_subplot(122)
    ax.imshow(atres, cmap="gray", vmin=0.0, vmax=1.0)
    plt.show()

