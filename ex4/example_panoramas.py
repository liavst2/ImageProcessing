import matplotlib.pyplot as plt
import numpy as np
import os

import sol4
import sol4_utils
import time

t1, t2 = None, None

def generate_panorama(data_dir, file_prefix, num_images, figsize=(20,20)):
    t1 = time.time()
    # The naming convention for a sequence of images is nameN.jpg, where N is a running number 1,2,..
    files = [os.path.join(data_dir,'%s%d.jpg'%(file_prefix, i+1)) for i in range(num_images)]

    # Read images.
    ims = [sol4_utils.read_image(f,1) for f in files]
    # Extract feature point locations and descriptors.
    p_d = [sol4.im_to_points(im) for im in ims]

    # Compute homographies between successive pairs of images.
    Hs = []
    for i in range(num_images-1):
        points1, points2 = p_d[i][0], p_d[i+1][0]
        desc1,   desc2   = p_d[i][1], p_d[i+1][1]

        # Find matching feature points.
        ind1, ind2 = sol4.match_features(desc1, desc2, .7)
        points1, points2 = points1[ind1,:], points2[ind2,:]

        # Compute homography using RANSAC.
        H12, inliers = sol4.ransac_homography(points1, points2, 5000, 6)

        # Display inlier and outlier matches.
        #sol4.display_matches(ims[i], ims[i+1], points1 , points2, inliers=inliers)
        # exit(0)
        Hs.append(H12)

    # Compute composite homographies from the panorama coordinate system.
    Htot = sol4.accumulate_homographies(Hs, (num_images-1)//2)

    # Final panorama is generated using 3 channels of the RGB images
    ims_rgb = [sol4_utils.read_image(f,2) for f in files]

    # Render panorama for each color channel and combine them.
    panorama = [sol4.render_panorama([im[...,i] for im in ims_rgb], Htot) for i in range(3)]
    panorama = np.dstack(panorama)

    #plot the panorama
    t2 = time.time()
    print("time: ", t2 - t1)
    plt.figure()
    plt.imshow(panorama.clip(0,1))
    plt.show()


def main():
    # generate_panorama('ext/', 'office'  , 4)
    # generate_panorama('ext/', 'backyard', 3)
    generate_panorama('ext/', 'oxford'  , 2)
    # generate_panorama('external/', 'levi', 3)


if __name__ == '__main__':
    main()
