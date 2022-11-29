# Hyperspectral Image Reconstruction - CSC2529 Project

## Simulated Images

Sample simulated image (using included RGB image as a hyperspectral image stand
        in) by -

    python amatrix.py

This uses the image formation matrix A in Afun and its transpose Atfun to sample
rows from each of the channels of the source image, to simulate capture of each
channel using a rolling shutter.

![Simulated rolling shutter image](afun.png?raw=true "Afun")


