#############################################################
# FILE: sol4_utils.py
# WRITER: Liav Steinberg
# EXERCISE : Image Processing ex4
#############################################################

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
from scipy import signal as sg
from scipy.misc import imread
from skimage.color import rgb2gray
from scipy.ndimage import filters as flt
import numpy as np



# --------------------------helpers-----------------------------#


def read_image(filename, representation):
    #######################################################
    # reads an image with the given representation
    #######################################################
    image = imread(filename).astype(np.float32) / 255
    return image if representation == 2 else rgb2gray(image)


def create_gauss_filter(k_size, dim):
    ########################################################
    # Helper function to calculate gaussian filter_vec
    ########################################################
    base = np.array([[1, 1]])
    filter_vec = np.array([[1, 1]])
    for i in range(2, k_size):
        filter_vec = sg.convolve2d(filter_vec, base).astype(np.float32)
    # normalize the filter
    filter_vec /= np.sum(filter_vec)
    return np.outer(filter_vec, filter_vec) if dim == 2 else filter_vec


def reduce(im, filt):
    #######################################################
    # shrinks image by a factor of 1/2
    #######################################################
    red = flt.convolve(flt.convolve(im, filt), filt.reshape(filt.size, 1))
    return red[::2, ::2]


def expand(im, filt):
    #######################################################
    # expand image by a factor of 2
    #######################################################
    exp = np.zeros((im.shape[0] * 2, im.shape[1] * 2), dtype=np.float32)
    exp[::2, ::2] = im[:, :]
    return flt.convolve(flt.convolve(exp, 2*filt), 2*filt.reshape(filt.size, 1))


def build_gaussian_pyramid(im, max_levels, filter_size):
    #######################################################
    # returns an array representing gaussian pyramid built
    # by a gaussian filter
    #######################################################
    gauss_pyr = [im]
    filter_vec = create_gauss_filter(filter_size, dim=1)
    for i in range(max_levels - 1):
        im = reduce(im, filter_vec)
        if im.shape[0] < 16 or im.shape[1] < 16:
            break
        gauss_pyr.append(im)
    return gauss_pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    #######################################################
    # returns an array representing laplacian pyramid built
    # by the algorithm from the tirgul
    #######################################################
    gauss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    lapl_pyr = []
    for i in range(max_levels - 1):
        curr = gauss_pyr[i]
        exp_curr = expand(gauss_pyr[i + 1], filter_vec)
        # check images sizes before subtracting
        if exp_curr.shape[0] > curr.shape[0]:
            exp_curr = np.delete(exp_curr, -1, axis=0)
        if exp_curr.shape[1] > curr.shape[1]:
            exp_curr = np.delete(exp_curr, -1, axis=1)
        # adding to the pyramid
        lapl_pyr.append(curr - exp_curr)
    # add the last gauss pyramid element
    lapl_pyr.append(gauss_pyr[-1])
    return lapl_pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    #######################################################
    # constructs the original image from its laplacian
    # pyramid
    #######################################################
    img = np.zeros_like(lpyr[-1])
    correct_shape = lpyr[0].shape
    for mat, co in zip(reversed(lpyr), reversed(coeff)):
        # adjustments before adding the matrices
        if img.shape[0] > mat.shape[0]:
            img = np.delete(img, -1, axis=0)
        if img.shape[1] > mat.shape[1]:
            img = np.delete(img, -1, axis=1)
        img += mat * co
        img = expand(img, filter_vec) if \
            img.shape != correct_shape else img
    return img


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    #######################################################
    # blends two images according to a given mask
    #######################################################
    L1, filt1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    L2, filt1 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    G, filt2 = build_gaussian_pyramid(mask.astype(np.float32), max_levels, filter_size_mask)
    Lout = []
    for i in range(max_levels):
        curr = (G[i] * L1[i]) + ((1.0 - G[i]) * L2[i])
        Lout.append(curr)
    return np.clip(laplacian_to_image(Lout, filt1, np.ones(len(Lout))), 0, 1)


def blur_spatial(im, kernel_size):
    #########################################################
    # implements gaussian blurring on a given image using a
    # gaussian filter calculated by approximating binomial
    # coefficients, and convolving it with the image
    #########################################################
    if kernel_size == 1:
        return im
    gauss_kernel = create_gauss_filter(kernel_size, dim=2)
    return sg.convolve2d(im, gauss_kernel, mode="same")


def least_squares_homography(points1, points2):
    #######################################################
    # taken from the additional files.
    #######################################################
    p1, p2 = points1, points2
    o0, o1 = np.zeros((p1.shape[0],1)), np.ones((p1.shape[0],1))

    A = np.vstack([np.hstack([p1[:,:1], o0, -p1[:,:1]*p2[:,:1], p1[:,1:],o0, -p1[:,1:]*p2[:,:1],o1,o0]),
                 np.hstack([o0, p1[:,:1],-p1[:,:1]*p2[:,1:],o0,p1[:,1:],-p1[:,1:]*p2[:,1:],o0,o1])])

    # Return None for unstable solutions
    if np.linalg.matrix_rank(A, 1e-3) < 8:
        return None
    if A.shape[0] == 8 and np.linalg.cond(A) > 1e10:
        return None

    H = np.linalg.lstsq(A, p2.T.flatten())[0]
    H = np.r_[H,1]
    return H.reshape((3,3)).T


def non_maximum_suppression(image):
    #######################################################
    # taken from the additional files.
    #######################################################
    # Find local maximas.
    neighborhood = generate_binary_structure(2,2)
    local_max = maximum_filter(image, footprint=neighborhood)==image
    local_max[image<(image.max()*0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num)+1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image).astype(np.bool)
    ret[centers[:,0], centers[:,1]] = True

    return ret


# --------------------------end-----------------------------#
