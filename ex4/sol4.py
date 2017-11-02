#############################################################
# FILE: sol4.py
# WRITER: Liav Steinberg
# EXERCISE : Image Processing ex4
#############################################################

from scipy.ndimage import map_coordinates
import sol4_utils as sut
import matplotlib.pyplot as plt
import scipy.signal as sg
import numpy as np
import itertools
import heapq


# --------------------------3.1-----------------------------#

def spread_out_corners(im, m, n, radius):
    #############################################################
    # takes from the additional files
    #############################################################
    corners = [np.empty((0,2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([y_bound[j], x_bound[i]])[np.newaxis,:]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:,0]>radius) & (corners[:,1]<im.shape[1]-radius) &
             (corners[:,1]>radius) & (corners[:,0]<im.shape[0]-radius))
    return corners[legit,:]


def harris_corner_detector(im):
    #############################################################
    # implements harris method for corner detection
    #############################################################
    dx = np.array([[1, 0, -1]])
    dy = dx.transpose()
    Ix = sg.convolve2d(im, dx, mode="same")
    Iy = sg.convolve2d(im, dy, mode="same")
    # blurring
    Ix_blur = sut.blur_spatial(Ix ** 2, 3)
    Iy_blur = sut.blur_spatial(Iy ** 2, 3)
    IxIy_blur = sut.blur_spatial(Ix * Iy, 3)
    # compute determinant and trace of M
    det = Ix_blur * Iy_blur - IxIy_blur ** 2
    tr = Ix_blur + Iy_blur
    R = det - 0.04 * (tr ** 2)
    return np.transpose(np.nonzero(sut.non_maximum_suppression(R)))


def sample_descriptor(im, pos, desc_rad):
    #############################################################
    # descriptor sampling
    #############################################################
    K = 1 + (2 * desc_rad)
    desc = np.zeros((K, K, pos.shape[0]), dtype=np.float32)
    for idx in range(len(pos)):
        x, y = pos[idx][0].astype(np.float32) / 4, pos[idx][1].astype(np.float32) / 4
        # map the coordinates
        X = np.arange(y - desc_rad, y + desc_rad + 1)
        Y = np.arange(x - desc_rad, x + desc_rad + 1)
        indices = np.transpose([np.tile(Y, len(X)), np.repeat(X, len(Y))])
        curr_desc = map_coordinates(im, [indices[:, 0],indices[:, 1]],order=1, prefilter=False).reshape(K, K)
        # normalize the descriptor
        E = np.mean(curr_desc)
        curr_desc = (curr_desc - E) / np.linalg.norm(curr_desc - E)
        desc[:, :, idx] = curr_desc
    return desc


def im_to_points(im):
    #############################################################
    # implements the function in example_panoramas.py
    #############################################################
    pyr, vec = sut.build_gaussian_pyramid(im, 3, 3)
    return find_features(pyr)


def find_features(pyr):
    #############################################################
    # finds features in an image given by its pyramid pyr
    #############################################################
    pos = spread_out_corners(pyr[0], 7, 7, 12)
    desc = sample_descriptor(pyr[2], pos, 3)
    return pos, desc


# --------------------------3.2-----------------------------#

def match_features(desc1, desc2, min_score):
    #############################################################
    # matches two descriptors taken from desc1, desc2 according
    # to some minimal score min_score
    #############################################################
    match_ind1, match_ind2 = [], []
    desc1_2nd, desc2_2nd = {}, {}
    # flatting the descriptors
    flat_desc1 = list(map(np.ravel, np.rollaxis(desc1, 2)))
    flat_desc2 = list(map(np.ravel, np.rollaxis(desc2, 2)))

    for (idx1, d1), (idx2, d2) in itertools.product(enumerate(flat_desc1),
                                                    enumerate(flat_desc2)):
        # filtering by the conditions
        # condition 1
        dot = np.inner(d1, d2)
        if not dot > min_score:
            continue
        # condition 2
        if idx1 not in desc1_2nd:
            desc1_2nd[idx1] = heapq.nlargest(2, np.dot(d1, np.transpose(flat_desc2)))[1]
        if not dot >= desc1_2nd[idx1]:
            continue
        # condition 3
        if idx2 not in desc2_2nd:
            desc2_2nd[idx2] = heapq.nlargest(2, np.dot(d2, np.transpose(flat_desc1)))[1]
        if not dot >= desc2_2nd[idx2]:
            continue
        # if they fulfill the conditions, they match
        match_ind1.append(idx1)
        match_ind2.append(idx2)

    return np.array(match_ind1), np.array(match_ind2)


# --------------------------3.3-----------------------------#

def apply_homography(pos1, H12):
    #############################################################
    # applying homographic transformation on given indexes
    #############################################################
    expand = np.column_stack((pos1, np.ones(len(pos1))))
    dot = np.dot(H12, expand.T).T
    normalized = (dot.T / dot[:,2]).T
    return np.delete(normalized, -1, axis=1)


def ransac_homography(pos1, pos2, num_iters, inlier_tol):
    #############################################################
    # applying RANSAC routine on the matches
    #############################################################
    pos1, pos2 = np.array(pos1), np.array(pos2)
    best_inliers = []
    for i in range(num_iters):
        # extract 4 random point and compute homography
        idx = np.random.random_integers(0, pos1.shape[0] - 1, size=4)
        points1, points2 = pos1[idx], pos2[idx]
        H12 = sut.least_squares_homography(points1, points2)
        # avoid unstable results
        if H12 is None:
            continue
        to_pos2 = np.array(apply_homography(pos1, H12))
        # compute amount of inliers
        in_indices = np.where(np.sum((to_pos2 - pos2)**2, axis=1) < inlier_tol)[0]
        best_inliers = in_indices if len(in_indices) > len(best_inliers) else best_inliers
    # recompute the homography
    points1, points2 = pos1[best_inliers], pos2[best_inliers]
    H12 = sut.least_squares_homography(points1, points2)
    return H12, np.array(best_inliers)


def display_matches(im1, im2, pos1, pos2, inliers):
    #############################################################
    # display the matches detected by RANSAC
    #############################################################
    pos1, pos2 = np.array(pos1), np.array(pos2)
    ins1, ins2 = pos1[inliers], pos2[inliers]
    outs1, outs2 = np.delete(pos1, inliers, axis=0), np.delete(pos2, inliers, axis=0)
    plt.figure()
    plt.imshow(np.hstack((im1, im2)), 'gray')
    plt.plot([ins1[:, 1], ins2[:, 1] + im1.shape[1]],
             [ins1[:, 0], ins2[:, 0]], mfc='r', c='y', lw=1.1, ms=5, marker='o')
    plt.plot([outs1[:, 1], outs2[:, 1] + im1.shape[1]],
             [outs1[:, 0], outs2[:, 0]], mfc='r', c='b', lw=.4, ms=5, marker='o')
    plt.show()

# --------------------------3.3-----------------------------#

def accumulate_homographies(H_successive, m):
    #############################################################
    # compute accumulated homographies between successive images
    #############################################################
    if not m:
        return [np.eye(3), np.linalg.inv(H_successive[0])]
    left_slice, right_slice = H_successive[:m], list(map(np.linalg.inv, H_successive[m:]))
    left_slice = list(itertools.accumulate(left_slice[::-1], np.dot))[::-1]
    right_slice = list(itertools.accumulate(right_slice, np.dot))
    left_slice.append(np.eye(3))
    H2m = np.array(left_slice + right_slice)
    H2m = (H2m.T / H2m[:,2,2]).T
    return H2m


# --------------------------4.3-----------------------------#

def prepare_panorama_base(ims, Hs):
    #############################################################
    # ims - the list of images, Hs - the list of homographies
    #############################################################
    corner_points = np.zeros((4, len(ims)))
    centers = np.zeros((len(ims), 2))
    for i in range(len(ims)):
        rows, cols = float(ims[i].shape[0]), float(ims[i].shape[1])
        corners = [[0, 0], [rows - 1, 0], [0, cols - 1], [rows - 1, cols - 1]]
        new_corners = np.array(apply_homography(corners, Hs[i]))
        corner_points[0, i] = np.max(new_corners[:, 0])
        corner_points[1, i] = np.min(new_corners[:, 0])
        corner_points[2, i] = np.max(new_corners[:, 1])
        corner_points[3, i] = np.min(new_corners[:, 1])
        centers[i,:] = np.array(apply_homography([[(rows-1)/2, (cols-1)/2]], Hs[i]))
    return corner_points, centers


def render_panorama(ims, Hs):
    #############################################################
    # rendering the panorama produced by the images ims
    #############################################################
    corners, centers = prepare_panorama_base(ims, Hs)
    Xmin, Xmax = np.min(corners[3,:]), np.max(corners[2,:])
    Ymin, Ymax = np.min(corners[1,:]), np.max(corners[0,:])
    Ypano, Xpano = np.meshgrid(np.arange(Xmin, Xmax+1), np.arange(Ymin, Ymax+1))
    panorama = np.zeros_like(Xpano)
    # calculate borders
    borders = [0]
    for i in range(len(ims) - 1):
        borders.append(np.round((centers[i,1]+centers[i+1,1])/2)-Xmin)
    borders.append(panorama.shape[1])
    # rendering
    for i in range(len(ims)):
        left_border, right_border = int(borders[i]), int(borders[i+1])
        X, Y = Xpano[:,left_border:right_border], Ypano[:,left_border:right_border]
        indices = np.array(apply_homography(np.transpose([X.ravel(), Y.ravel()]), np.linalg.inv(Hs[i])))
        strip = panorama[:, left_border:right_border]
        image = map_coordinates(ims[i], [indices[:, 0], indices[:,1]], order=1, prefilter=False)
        panorama[:, left_border:right_border] = image.reshape(strip.shape)

    ### ATTENTION ###
    # I intentionally used this separated for loop for the blending part. The reason
    # is that the resulting panorama after this block of code is more highlighted on
    # the borders rather than if I return the panorama without the blending. So I split
    # the blending from the rest of the code so you can comment it comfortably and see
    # the difference yourself (if you want:) )
    # I could not figure out the problem.

    for i in range(len(ims)-1):
        border = int(borders[i+1])
        mask_strip = np.zeros_like(panorama)
        mask_strip[:,border:] = np.ones_like(mask_strip[:,border:])
        left_im = panorama[:,:border]
        right_im = panorama[:,border:]
        left_im_pano = np.hstack((left_im, np.zeros((right_im.shape[0], right_im.shape[1]))))
        right_im_pano = np.hstack((np.zeros((left_im.shape[0], left_im.shape[1])), right_im))
        panorama = sut.pyramid_blending(right_im_pano, left_im_pano, mask_strip, 4, 31, 31)

    return panorama

# --------------------------end-----------------------------#
