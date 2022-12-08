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

def Atfun(x, channel=0, num_channels=3, extra=0):
    '''
    Calculates Atx where At is the transpose of the image formation matrix A,
    and x is an input image.

    Inputs:
    x: Image of size (height, width)
    channel: Which channel number

    Outputs:
    return: New image of size (height, width, channels)
    '''

    # Create mask and transpose
    mask = make_mask(x.shape, channel+channel*extra, extra, num_channels)
    mask = np.sum(mask, axis=0)

    # Sum mask with x
    return mask * x


def Afun(x, channel=0, num_channels=3, extra=0):
    '''
    Calculates Ax where A is the image formation matrix, and x is an input
    image.

    Inputs:
    x: Image of size (height, width, d) with d channels

    Outputs:
    return: New image of size (height, width) sampled from the d channels of x
    '''

    # Create mask
    mask = make_mask(x.shape, channel+channel*extra, extra, num_channels)

    # Mupltiply each image with its corresponding mask, then sum all masked
    # images into final image.
    return mask * x


def SimulateImage(img, num_channels, sigma=0.001, e=0):
    '''
    Helper function to create a simulated image
    '''

    # Create a mask for each channel and stack them together
    channels = [Afun(img[:,:,c], c, num_channels, e)
            for c in range(num_channels)]
    channels = np.stack(channels, axis=2)

    # Add noise
    channels += sigma*np.random.randn(img.shape[0], img.shape[1], num_channels)

    return channels


if __name__=="__main__":

    img = io.imread('birds.png').astype(float)/255.

    img = img[:,:,0]

    c = 0
    n = 3
    e = int(img.shape[1]/n)

    for c in range(3):

        ares = Afun(img, c, n, e)
        atres = Atfun(ares, c, n, e)

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(ares, cmap="gray", vmin=0.0, vmax=1.0)
        ax = fig.add_subplot(122)
        ax.imshow(atres, cmap="gray", vmin=0.0, vmax=1.0)
        plt.show()

