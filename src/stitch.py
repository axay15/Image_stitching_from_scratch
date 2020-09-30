"""


This program performs image stitching for more than two images.
This code will work only with ".jpg" files

I have referred to two links to complete this project :

1. http://slazebni.cs.illinois.edu/spring18/assignment3.html
Ideas borrowed:
    1. scipy.spatial.distance.cdist(X,Y,'sqeuclidean') for fast computation of Euclidean distance.
    2. Homomgraphies can be computed using SVD
    3. np.linalg.norm for calculating inliers

2. http://www.cim.mcgill.ca/~langer/558/2009/lecture19.pdf
Ideas borrowed:
    1. how to write 'A' matrix for solving homography using SVD
"""



import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
import argparse
import os
from scipy.spatial.distance import cdist
import random
import copy
import sys


def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def compute_SIFT(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    return kp1, des1, kp2, des2

def compute_match_euclidean(kp1, des1, kp2, des2):

    dist = cdist(des1, des2, 'sqeuclidean')                                 # compute the euclidean distance between the points
    # a = np.where(pts_distances <5000)[1]                                  # trial and error on threshold
    # print(a)

    pts_within_threshold = np.where(dist < 7000)

    img1_coordinates = np.array([kp1[pts].pt for pts in pts_within_threshold[0]])
    img2_coordinates = np.array([kp2[pts].pt for pts in pts_within_threshold[1]])

    return np.concatenate((img1_coordinates, img2_coordinates), axis=1)


def compute_homography(random_points_4):

    pt1_img1 = random_points_4[0, 0:2]
    pt2_img1 = random_points_4[1, 0:2]
    pt3_img1 = random_points_4[2, 0:2]
    pt4_img1 = random_points_4[3, 0:2]

    pt1_img2 = random_points_4[0, 2:4]
    pt2_img2 = random_points_4[1, 2:4]
    pt3_img2 = random_points_4[2, 2:4]
    pt4_img2 = random_points_4[3, 2:4]

    P_1 = [pt1_img1, pt2_img1, pt3_img1, pt4_img1]
    P_2 = [pt1_img2, pt2_img2, pt3_img2, pt4_img2]

    A = []

    # http://www.cim.mcgill.ca/~langer/558/2009/lecture19.pdf
    # referred to above link to understand how to create the A matrix and solve for homography using SVD

    for i in range(len(P_1)):
        r1 = [-P_1[i][0], -P_1[i][1], -1, 0, 0, 0, P_1[i][0] * P_2[i][0], P_1[i][1] * P_2[i][0], P_2[i][0]]
        r2 = [0, 0, 0, -P_1[i][0], -P_1[i][1], -1, P_1[i][0] * P_2[i][1], P_1[i][1] * P_2[i][1], P_2[i][1]]

        A.append(r1)
        A.append(r2)

    A = np.matrix(A)
    # print (A)

    # compute svd to find homography
    u, sig, v = np.linalg.svd(A)
    H = v[len(v) - 1]  # last column is the homography matrix
    H = H.reshape(3, 3)  # 3x3 Homography matrix
    H = H / H[2, 2]  # so that the last element (H[2,2]) will be 1
    return H


def ransac(match_pts):
    threshold = 0.6
    max_inliers = 0
    counter = 0
    iterations = 2000           # run for 2000 iterations to get the maximum inliers
    while (counter < iterations):

        # generate 4 random points to compute homography
        random_indexes = random.choices(range(len(match_pts)), k=4)
        random_points_4 = match_pts[random_indexes]

        #
        #
        # computing homography using SVD . 8 equations required atleast..
        H = compute_homography(random_points_4)

        # find new set of points if homography matrix has a rank <3
        if np.linalg.matrix_rank(H) < 3:
            continue

        # calculate number of inliers using || p'-Hp || < threshold:

        p = np.insert(match_pts[:, 0:2], 2, 1.0, axis=1)
        p_dash = match_pts[:, 2:4]

        calc = []
        for i in range(len(match_pts)):
            r = np.dot(H, p[i])
            r = r / r[0, 2]  # to get answer of the form [x,y,1]; normalize with the last element
            calc.append(r)

        temp = np.asarray(calc)
        Hp = temp[:, 0][:, 0:2]

        ssd = np.linalg.norm(p_dash - Hp, axis=1) ** 2

        inlier_indices = np.where(ssd < 0.5)

        tot_inliers = len(inlier_indices[0])

        if tot_inliers > max_inliers:
            max_inliers = tot_inliers
            inliers = match_pts[inlier_indices]
            true_H = H

        counter += 1
        # for 2000 iterations

    # print(max_inliers)
    return true_H, max_inliers


def is_stitch_possible(img1, img2, det=False):
    """
    return first element as 0 if stitching not possible between the images, else return first element as 1.
    if the first element is 1, then return the homography, length of matches and the correct order of the images to be
    stitched. This function also determines the order in which the images are to be stitched
    """
    result = [0]
    #kp1, des1, kp2, des2 = compute_SIFT(img1, img2)
    try:
        kp1, des1, kp2, des2 = compute_SIFT(img1, img2)
        match_pts = compute_match_euclidean(kp1, des1, kp2, des2)
    except:
        result.extend((0, 0, img1, img2))
        return result
    else:
        h, out_cnt = ransac(match_pts)
        #print(out_cnt)

        if out_cnt < 6:  # very few inliers (when there's very few matches) mean these images can't be stitched
            result.extend((0, len(match_pts), img1, img2))
            return result

        elif h[0,2] < 0 or h[1, 2] < 0:   # if the last column contains negative values, switch the order of images
            result[0] = 1
            img1, img2 = img2, img1           # swap the order of images to perform left warping
            kp1, des1, kp2, des2 = compute_SIFT(img1, img2)
            match_pts = compute_match_euclidean(kp1, des1, kp2, des2)
            h, out_cnt = ransac(match_pts)          # calculate homography again and return
            result.extend((h, len(match_pts), img1, img2))
            return result

        else:
            result[0] = 1
            if det:
                result.extend((h, len(match_pts), img1, img2))

            return result


def create_panorama(img1, img2, homography_mat):
    panaroma_image1 = cv2.warpPerspective(img1, homography_mat,
                                          (int(img1.shape[1] + img2.shape[1]*0.9),
                                           int(img1.shape[0] + img2.shape[0]*0.5)))
    # pan = cv2.warpPerspective(img1, ran, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    panaroma_image1[0:img2.shape[0], 0:img2.shape[1]] = img2

   

    return panaroma_image1

def remove_black_pixels(image):
    """
    Recursively call this function to remove black pixels from all 4 sides of the image
    """

    # black pixels will sum to zero
    if np.sum(image[0]) == 0:                           # remove black pixels from first row
        return remove_black_pixels(image[1:])
    if np.sum(image[-1]) == 0:                          # remove black pixels from last row
        return remove_black_pixels(image[:-2])
    if np.sum(image[:,0]) == 0:                         # remove black pixels from first column
        return remove_black_pixels(image[:,1:])
    if np.sum(image[:,-1]) == 0:                        # remove black pixels from last column
        return remove_black_pixels(image[:,:-2])
    return image


################## MAIN #######################

sys.setrecursionlimit(10000)             # python doesn't provide a good suppor for recursion; there's a limit

if len(sys.argv) < 2:
    print("No directory provided")
    exit(0)

dir = sys.argv[1]

if (not os.path.exists(dir)):
    print("No such directory")
    exit(0)

images = []


# read all the .jpg files
for image in os.listdir(dir):
    if '.jpg' in image:
        image_full_path = os.path.join(dir, image)
        images.append(cv2.imread(image_full_path))

if len(images) < 2:
    print("Not enough 'JPG' images available to perform stitching")
    exit(0)

if len(images) == 2:
    result = is_stitch_possible(images[0], images[1], True)
    if result[0] == 0:
        print("Stitch not possible between these images")
        exit(0)

    else:
        result = is_stitch_possible(images[0], images[1], True)
        images[0], images[1] = result[-2], result[-1]                     # get the correct order of images to stitch
        pan = create_panorama(images[0], images[1], result[1])
        pan = remove_black_pixels(pan)
        cv2.imwrite(dir + "/panorama.jpg", pan)

elif len(images) == 3:
    #kp1, des1, kp2, des2 = compute_SIFT(images[0], images[1])

    # find matches in all pairs of images
    img12 = is_stitch_possible(images[0], images[1], True)
    img13 = is_stitch_possible(images[0], images[2], True)
    img23 = is_stitch_possible(images[1], images[2], True)

    # if the first element is 0 for the above function calls, images cannot be stitched

    #print(img12)
    #print(img13)
    #print(img23)

    possible_images = []

    # append images that can be stitched.

    if img12[0] ==1:
        possible_images.append(image[0])
        possible_images.append(image[1])

    if img13[0] == 1:
        possible_images.append(image[0])
        possible_images.append(image[2])

    if img23[0] ==1:
        possible_images.append(image[1])
        possible_images.append(image[2])

    possible_images = list(set(possible_images))

    if len(possible_images) == 2:
        result = is_stitch_possible(possible_images[0], possible_images[1], True)
        possible_images[0], possible_images[1] = result[-2], result[-1]
        pan = create_panorama(possible_images[0], possible_images[1], result[1])

    elif len(possible_images) == 3:
        img1_matches = img12[2] + img13[2]
        img2_matches = img12[2] + img23[2]
        img3_matches = img13[2] + img23[2]

        if img1_matches == max(img1_matches,img2_matches,img3_matches):
            possible_images[1] = images[0]
            possible_images[0] = images[1]
            possible_images[2] = images[2]

        elif img3_matches == max(img1_matches,img2_matches,img3_matches):
            possible_images[1] = images[2]
            possible_images[0] = images[0]
            possible_images[2] = images[1]

        else:
            possible_images[1] = images[1]
            possible_images[0] = images[0]
            possible_images[2] = images[2]


        # stitch the first two images and create a temporary panorama image
        result = is_stitch_possible(possible_images[0], possible_images[1], True)
        possible_images[0], possible_images[1] = result[-2], result[-1]
        pan1 = create_panorama(possible_images[0], possible_images[1], result[1])
        pan1 = remove_black_pixels(pan1)
        cv2.imwrite(dir + "/panorama1.jpg", pan1)
        pan1 = cv2.imread(dir + "/panorama1.jpg")

        # read the temporary panorama and stitch it with the third image
        result = is_stitch_possible(pan1, possible_images[2], True)
        pan1, possible_images[2] = result[-2], result[-1]
        pan2 = create_panorama(pan1, possible_images[2], result[1])
        pan = remove_black_pixels(pan2)
        cv2.imwrite(dir + "/panorama.jpg", pan)
        os.remove(dir + '/panorama1.jpg')
    


elif len(images) > 3:

    stitched_images = []
    pan = images[0]    # assume the first image is in panorama

    for i in range(len(images)):  # start with some image
        for j in range(len(images)):  # check if the previous image can be stitched with any of the images
            if i != j:                # so that both the loops dont look at the same image
                if j not in stitched_images:   # if the image has already been stitched, ignore it; look for the next

                    result = is_stitch_possible(pan, images[j], True)
                    if result[0] == 1:              # if stitch possible, create a panorama
                        pan, img = result[-2], result[-1]
                        pan = create_panorama(pan, img, result[1])
                        pan = remove_black_pixels(pan)
                        cv2.imwrite(dir + "/temp.jpg", pan)

                        pan = cv2.imread(dir + "/temp.jpg")
                        stitched_images.append(j)
                        if len(stitched_images) == 1:
                            stitched_images.append(i)
                        break                            # if a panorama has been created, break; look for next image

        if pan.shape == images[i].shape and i < len(images)-1:
            pan = images[i+1]           # the image we started with is not a part of panorama; start with the next image

    pan = remove_black_pixels(pan)
    cv2.imwrite(dir + "/panorama.jpg", pan)
    os.remove(dir + '/temp.jpg')
