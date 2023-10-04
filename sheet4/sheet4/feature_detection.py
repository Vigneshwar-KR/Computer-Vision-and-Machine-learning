#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal as sig

from utils import *


# Task 1
#
# Implement the Harris-Stephens Corner Detection for `imgGray1` without using an existing all-in-one function, 
# e.g. do not use functions like `cv2.cornerHarris(..)`.

img1 = cv2.imread('C:/Vicky/Tu Braunschweig/Semester 2/Computer vision/Exercise/Team13/sheet4/sheet4/img/building.jpeg')
img1Gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
showImage(img1Gray)



# First, you need to create the Harris matrix.

# TODO: Use the Sobel filter (with `ksize`) to get x and y derivatives of `img1Gray`.
""" ksize = 5

def gradient_x(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    G_x = sig.convolve2d(image, sobel_x, mode='same')
    return G_x
def gradient_y(image):
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    G_y= sig.convolve2d(image, sobel_y, mode='same')
    return G_y

gradientX = gradient_x(img1Gray) # Sobel gradients in x (Gx)
gradientY = gradient_y(img1Gray) # Sobel gradients in y (Gy) """


ksize = 5
gradientX = cv2.Sobel(img1Gray, cv2.CV_64F, 1, 0, ksize=ksize)
gradientY = cv2.Sobel(img1Gray, cv2.CV_64F, 0, 1, ksize=ksize)


# TODO: Create a simple box filter smoothing kernel (use `ksize` again).
#
def box_filter(ksize):
    kernal = np.ones((ksize, ksize)) / (ksize * ksize)
    return kernal 

smoothingKernel = box_filter(ksize)

#

# TODO: Compute and fill in the values of the Harris matrix from the Sobel gradients.
harrisMatrix = []
harrisMatrix = np.ones((2, 2) + img1Gray.shape)

#  Hint: Each of the following 4 entries contains a full gradient image
harrisMatrix[0, 0] = gradientX * gradientX # Gx^2
harrisMatrix[0, 1] = gradientX * gradientY # Gx*Gy
harrisMatrix[1, 0] = gradientX * gradientY # Gx*Gy
harrisMatrix[1, 1] = gradientY * gradientY # Gy^2
#
#print(harrisMatrix)
#

# TODO: Use the created smoothing kernel to filter the 4 Harris matrix values assigned above.
#  Tipp: You can use `cv2.filter2D(..)` to apply a kernel to a whole image.
#

for i in range(2):
    for j in range(2):
        harrisMatrix[i, j] = cv2.filter2D(harrisMatrix[i, j], -1, smoothingKernel)


# TODO: Calculate the Harris-Stephens score (R) for each pixel.
#  Tipp: Make sure you find and use functions for the intermediate steps that are available in OpenCV.
harris_k = .05 # Empirical k value
R = np.ones(img1Gray.shape)
for x in range(R.shape[0]):
    for y in range(R.shape[1]):
        H = harrisMatrix[:, :, x, y] # Get H for the current pixel
        det = np.linalg.det(H)
        trace = np.trace(H)
        R[x, y] = det - harris_k * (trace ** 2) # det(H) - harris_k * trace(H)^2
        #
harris_r_norm = cv2.normalize(R, None, 0, 1, norm_type=cv2.NORM_MINMAX) # Normalize to 0-1 for display and thresholding


# TODO: Select pixels with a relevant Harris-Stephens score and highlight these in `imgMarkers` using `cv2.drawMarker(..)`
harris_tau = .5 # Harris-Stephens score threshold
imgMarkers = img1.copy()

# Select pixels with a relevant Harris-Stephens score above the threshold
# Highlight the recognized corner positions in imgMarkers
relevant_pixel = np.argwhere(harris_r_norm > harris_tau)

# Highlight the recognized corner positions in imgMarkers
for position in relevant_pixel:
    x, y = position[0], position[1]
    cv2.drawMarker(imgMarkers, (y, x), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=2)

plt.figure(figsize=(10, 3))
showImages([("Input", img1), ("Harris-Stephens score (R)", harris_r_norm), ("Corners", imgMarkers)])




#
# Task 2
#
# Use the SIFT Feature detector to find matching features in two images, in order to create a combined panorama image.

img2 = cv2.imread('C:/Vicky/Tu Braunschweig/Semester 2/Computer vision/Exercise/Team13/sheet4/sheet4/img/mountain1.png')
img3 = cv2.imread('C:/Vicky/Tu Braunschweig/Semester 2/Computer vision/Exercise/Team13/sheet4/sheet4/img/mountain2.png')

# convert to greyscale
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

# TODO: Extract SIFT keypoints (`kp1`, `kp1`) and feature descriptors (`fd1`, `fd2`) for both images (`img2`, `img3`).
#  (https://docs.opencv.org/master/da/df5/tutorial_py_sift_intro.html)

# create SIFT feature extractor
sift = cv2.SIFT_create()

kp1, fd1 = sift.detectAndCompute(img2, None)
kp2, fd2 = sift.detectAndCompute(img3, None) 


# TODO: For all features of `img2`, find the two closest matches from the features of `img3` using euclidean distances.
#  Tip: Have a look at `knnMatch(..)` from `cv2.BFMatcher`.
# Create a BFMatcher object
bf = cv2.BFMatcher()

# Find two closest matches for each feature of img2 from img3 using Euclidean distances
matches = bf.knnMatch(fd1, fd2, k=2)
# The value of k specifies the number of nearest neighbors to return. 
# In this case, we set it to 2 to find the two closest matches for each feature.


# TODO: Use the ratio test (best vs. second-best match) to keep only the `good_matches`.
best_to_secondBest_ratio = .6
good_matches = []
for m, n in matches:
    if m.distance < best_to_secondBest_ratio * n.distance:
        good_matches.append(m)


# TODO: Create an image showing the matches between `img2` and `img3`.
imgMatches = cv2.drawMatches(img2, kp1, img3, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)



# TODO: Change this, once you have completed task 2 to test your feature matches.
task2_complete = True
#
# ???
#

if not task2_complete:
    showImages([("img2", img2), ("img3", img3)])
else:
    # Now let's try to stich these two images together to see how well the featues actually are.
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)
    # Apply transformation to transform `img3` onto `img2`.
    h, w, _ = img2.shape
    img23 = cv2.warpPerspective(img3, H, (w * 2, h))
    # Fill in pixels from `img2` around transformed `img3`.
    stitchempty = np.where(img23[:, :w, :] == [0,0,0])
    img23[stitchempty] = img2[stitchempty]

    plt.figure(figsize=(10, 5))
    showImages([("img2", img2), ("img3", img3), ("Matches", imgMatches), ("Both images stiched together", img23)], 2)