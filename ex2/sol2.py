#############################################################
# FILE: sol2.py
# WRITER: Liav Steinberg
# EXERCISE : Image Processing ex2
#############################################################


import numpy as np
from scipy.signal import convolve2d
from scipy.misc import imread
from matplotlib import pyplot as plt
from skimage.color import rgb2gray


#--------------------------helpers-----------------------------#

def read_image(filename, representation):
    #######################################################
    # reads an image with the given representation
    #######################################################
    image = imread(filename).astype(np.float32) / 255
    return image if representation == 2 else rgb2gray(image)


def transform_matrix(size, action):
    ########################################################
    # Helper function to calculate the dft / idft matrices
    ########################################################
    omega = np.exp(-2 * np.pi * 1J / size) if action == "dft"\
    	else np.exp(2 * np.pi * 1J / size)
    index1, index2 = np.meshgrid(np.arange(size), np.arange(size))
    mat = np.power(omega, index1 * index2)
    return mat.astype(np.complex128)


def create_gauss_kernel(k_size):
    ########################################################
    # Helper function to calculate gaussian kernel
    ########################################################
    base = kernel = np.array([[1, 1]])
    for i in range(2, k_size):
        kernel = convolve2d(kernel, base).astype(np.float32)
    gauss_kernel = np.dot(kernel.reshape(k_size, 1), kernel).astype(np.float32)
    # normalize so that coefficients will sum up to 1
    gauss_kernel /= np.sum(gauss_kernel)
    return gauss_kernel


#--------------------------1.1-----------------------------#

def DFT(signal):
    ########################################################
    # Performs DFT on a given signal
    ########################################################
    DFT_left_matrix = transform_matrix(signal.shape[0], "dft")
    return np.dot(DFT_left_matrix, signal)


def IDFT(fourier_signal):
    ########################################################
    # Performs inverted DFT on a given signal
    ########################################################
    IDFT_left_matrix = transform_matrix(fourier_signal.shape[0], "idft")
    return np.dot(IDFT_left_matrix, fourier_signal) / fourier_signal.shape[0]

#--------------------------1.2-----------------------------#

def DFT2(image):
    ########################################################
    # Performs DFT on a given image (2d matrix)
    ########################################################
    DFT_right_matrix = transform_matrix(image.shape[1], "dft")
    return np.dot(DFT(image), DFT_right_matrix)


def IDFT2(fourier_image):
    ########################################################
    # Performs IDFT on a given image (2d matrix)
    ########################################################
    IDFT_right_matrix = transform_matrix(fourier_image.shape[1], "idft")
    return np.dot(IDFT(fourier_image), IDFT_right_matrix) / fourier_image.shape[1]

#--------------------------2.1-----------------------------#

def conv_der(im):
    #########################################################
    # calculates the magnitude of the derivatives of a given
    # image using convolutions with the apt kernels
    #########################################################
    dx = np.array([[1, 0, -1]])
    dy = dx.reshape(3, 1)
    x_der = convolve2d(im, dx, mode="same")
    y_der = convolve2d(im, dy, mode="same")
    return np.sqrt(np.power(x_der, 2) + np.power(y_der, 2))

#--------------------------2.2-----------------------------#

def fourier_der(im):
    #########################################################
    # calculates the magnitude of the derivatives of a given
    # image using fourier transform
    #########################################################
    # shifting coefficients properly
    rows = np.arange(-im.shape[0] / 2, im.shape[0] / 2) if not im.shape[0] % 2\
        else np.arange(-im.shape[0] / 2, im.shape[0] / 2 - 1)
    cols = np.arange(-im.shape[1] / 2, im.shape[1] / 2) if not im.shape[1] % 2\
        else np.arange(-im.shape[1] / 2, im.shape[1] / 2 - 1)
    #calculating the derivatives
    u, v = np.meshgrid(cols, rows)
    x_der = 2 * np.pi * 1J * IDFT2(u * np.fft.fftshift(DFT2(im))) / im.size
    y_der = 2 * np.pi * 1J * IDFT2(v * np.fft.fftshift(DFT2(im))) / im.size
    return np.sqrt(np.abs(x_der)**2 + np.abs(y_der)**2)

#--------------------------3.1-----------------------------#

def blur_spatial(im, kernel_size):
    #########################################################
    # implements gaussian blurring on a given image using a
    # gaussian filter calculated by approximating binomial
    # coefficients, and convolving it with the image
    #########################################################
    if kernel_size == 1:
        return im
    gauss_kernel = create_gauss_kernel(kernel_size)
    return convolve2d(im, gauss_kernel, mode="same")

#--------------------------3.2-----------------------------#

def blur_fourier(im, kernel_size):
    #########################################################
    # implements gaussian blurring on a given image using a
    # gaussian filter calculated by approximating binomial
    # coefficients, and calculating the blur using their
    # fourier transforms
    #########################################################

    if kernel_size == 1:
        return im
    y_dim = im.shape[0]
    x_dim = im.shape[1]
    y_cen = int(np.floor(float(y_dim) / 2) + 1)
    x_cen = int(np.floor(float(x_dim) / 2) + 1)
    kernel_cen = int(np.floor(float(kernel_size) / 2) + 1)
    # calculating how much padding is needed
    padding = ((y_cen-kernel_cen+1, y_dim-y_cen-kernel_cen),
               (x_cen-kernel_cen+1, x_dim-x_cen-kernel_cen))
    # pad the kernel with zeros so it match in size with the image
    # and shift it to the center of the image
    gauss_kernel = create_gauss_kernel(kernel_size)
    pad_kernel = np.pad(gauss_kernel, padding, "constant", constant_values=(int(0),))
    pad_kernel = np.fft.ifftshift(pad_kernel)
    # calculating fourier transforms
    fourier_im = DFT2(im).astype(np.complex128)
    fourier_kernel = DFT2(pad_kernel).astype(np.complex128)
    # calculate the blur using the transforms
    blur_im = np.real(IDFT2(fourier_im * fourier_kernel))
    return blur_im.astype(np.float32)

#----------------------------end------------------------------#
