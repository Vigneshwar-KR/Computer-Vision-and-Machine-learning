#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

from flow_utils import *
from utils import *


# Task 2
#
# Implement Lucas-Kanade or Horn-Schunck Optical Flow.



# TODO: Implement Lucas-Kanade Optical Flow.
#
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# kernel_size: kernel size
# eigen_threshold: threshold for determining if the optical flow is valid when performing Lucas-Kanade
# returns the Optical flow based on the Lucas-Kanade algorithm
def LucasKanadeFlow(frames, Ix, Iy, It, kernel_size, eigen_threshold = 0.01):
    

    """
    #VICKY
    height, width = frames[0].shape
    flow = np.zeros((height, width, 2), dtype=np.float32)

    kernel = np.ones((kernel_size, kernel_size))
    

    for y in range(kernel_size // 2, height - kernel_size // 2):
        for x in range(kernel_size // 2, width - kernel_size // 2):
            # Extract the patches from the frames and derivatives
            Ix_patch = Ix[y - kernel_size // 2:y + kernel_size // 2 + 1, x - kernel_size // 2:x + kernel_size // 2 + 1]
            Iy_patch = Iy[y - kernel_size // 2:y + kernel_size // 2 + 1, x - kernel_size // 2:x + kernel_size // 2 + 1]
            It_patch = It[y - kernel_size // 2:y + kernel_size // 2 + 1, x - kernel_size // 2:x + kernel_size // 2 + 1]

            Ix_patch = Ix_patch.flatten()
            Iy_patch = Iy_patch.flatten()
            It_patch = It_patch.flatten()
           
            A = np.vstack((Ix_patch, Iy_patch)).T
            b = -It_patch
           
            # Compute the optical flow
            if np.min(np.linalg.eigvals(A.T @ A)) < eigen_threshold:
                continue

            flow_vector = np.linalg.pinv(A) @ b

            flow[y, x] = flow_vector
            #print(flow)
    return flow
    #return PLACEHOLDER_FLOW(frames)
    
    """
    

    #CHUNG
    height = frames[0].shape[0]
    width = frames[0].shape[1]
    

    flow = np.zeros((height, width, 2))

    for i in range(height):
        for j in range(width):
            # print(kernel_size)
            if ((i + 1) % kernel_size[1]) == 0 and ((j + 1) % kernel_size[0]) == 0:
                local_Ix = np.copy(Ix[i-kernel_size[1] + 1: i + 1, j - kernel_size[0] + 1:j + 1])
                local_Iy = np.copy(Iy[i-kernel_size[1] + 1: i + 1, j - kernel_size[0] + 1:j + 1])
                local_It = np.copy(It[i-kernel_size[1] + 1: i + 1, j - kernel_size[0] + 1:j + 1])

                Ixx = local_Ix * local_Ix
                Iyy = local_Iy * local_Iy
                Ixy = local_Ix * local_Iy
                Ixt = -local_Ix * local_It
                Iyt = -local_Iy * local_It

                structure_tensor = np.array([[np.sum(Ixx), np.sum(Ixy)], [np.sum(Ixy), np.sum(Iyy)]])
                t = np.array([[np.sum(Ixt)],
                              [np.sum(Iyt)]])
                eigenvalues, eigenvectors = np.linalg.eig(structure_tensor)
                if eigenvalues[0] > 0.01 and eigenvalues[1] > 0.01:
                    structure_tensor_inv = np.linalg.inv(structure_tensor)
                    local_flow = np.matmul(structure_tensor_inv, t)
                    flow[i-kernel_size[1] + 1: i + 1, j - kernel_size[0] + 1:j + 1, 0] = local_flow[0]
                    flow[i-kernel_size[1] + 1: i + 1, j - kernel_size[0] + 1:j + 1, 1] = local_flow[1]

    return flow


    #
    # ???
    #



# TODO: Implement Horn-Schunck Optical Flow.
#
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# window_size: the number of points taken in the neighborhood of each pixel
# max_iterations: maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
# epsilon: the stopping criterion for the difference when performing the Horn-Schuck algorithm
# returns the Optical flow based on the Horn-Schunck algorithm
def HornSchunckFlow(frames, Ix, Iy, It, max_iterations = 1000, epsilon = 0.002):
    return PLACEHOLDER_FLOW(frames)
    #
    # ???
    #



# Load image frames
frames = [  cv2.imread("C:/Vicky/Tu Braunschweig/Semester 2/Computer vision/Exercise/Team13/sheet5/sheet5/resources/frame1.png"),
            cv2.imread("C:/Vicky/Tu Braunschweig/Semester 2/Computer vision/Exercise/Team13/sheet5/sheet5/resources/frame2.png")]


# Load ground truth flow data for evaluation
flow_gt = load_FLO_file("C:/Vicky/Tu Braunschweig/Semester 2/Computer vision/Exercise/Team13/sheet5/sheet5/resources/groundTruthOF.flo")

# Grayscales
gray = [(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0).astype(np.float64) for frame in frames]

# Get derivatives in X and Y
xdk = np.array([[-1.0, 1.0],[-1.0, 1.0]])
ydk = xdk.T
fx =  cv2.filter2D(gray[0], cv2.CV_64F, xdk) + cv2.filter2D(gray[1], cv2.CV_64F, xdk)
fy = cv2.filter2D(gray[0], cv2.CV_64F, ydk) + cv2.filter2D(gray[1], cv2.CV_64F, ydk)

# Get time derivative in time (frame1 -> frame2)
tdk1 =  np.ones((2,2))
tdk2 = tdk1 * -1
ft = cv2.filter2D(gray[0], cv2.CV_64F, tdk2) + cv2.filter2D(gray[1], cv2.CV_64F, tdk1)

# Ground truth flow
plt.figure(figsize=(5, 8))
showImages([("Groundtruth flow", flowMapToBGR(flow_gt)),
            ("Groundtruth field", drawArrows(frames[0], flow_gt)) ], 1, False)

# Lucas-Kanade flow
flow_lk = LucasKanadeFlow(gray, fx, fy, ft, [15,15])#chung
#flow_lk = LucasKanadeFlow(gray, fx, fy, ft, 15)#vicky
error_lk = calculateAngularError(flow_lk, flow_gt)
plt.figure(figsize=(5, 8))
showImages([("LK flow - angular error: %.3f" % error_lk, flowMapToBGR(flow_lk)),
            ("LK field", drawArrows(frames[0], flow_lk)) ], 1, False)

# Horn-Schunk flow
"""flow_hs = HornSchunckFlow(gray, fx, fy, ft)
error_hs = calculateAngularError(flow_hs, flow_gt)
plt.figure(figsize=(5, 8))
showImages([("HS flow - angular error %.3f" % error_hs, flowMapToBGR(flow_hs)),
            ("HS field", drawArrows(frames[0], flow_hs)) ], 1, False)
            """

plt.show()
