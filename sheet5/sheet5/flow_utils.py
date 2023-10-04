
import os
import numpy as np
import cv2



def PLACEHOLDER_FLOW(frames):
    return np.array([[[x, y] for x in np.linspace(-1, 1, frames[0].shape[1])] for y in np.linspace(-1, 1, frames[0].shape[0])])

PLACEHOLDER_FLOW_VISUALIZATION = cv2.imread('C:/Vicky/Tu Braunschweig/Semester 2/Computer vision/Exercise/Team13/sheet5/sheet5/resources/example_flow_visualization.png')



#
# Task 1
#
# Implement utility functions for flow visualization.



# TODO: Convert a flow map to a BGR image for visualisation.
#       A flow map is a 2-channel 2D image with channel 1 and 2 depicting the portion flow in X and Y direction respectively.
def flowMapToBGR(flow_map):
    # Flow vector (X, Y) to angle and magnitudes
    
    hsv = np.zeros((flow_map.shape[0], flow_map.shape[1], 3), dtype=np.uint8)
    hsv[..., 2] = 255
    mag, ang = cv2.cartToPolar(flow_map[..., 0], flow_map[..., 1],3)


    # Angle and vector size to HSV color
    # Use Hue and Value to encode the Optical Flow
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
 
    # Convert HSV image into BGR for demo
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #print(PLACEHOLDER_FLOW_VISUALIZATION)

    return bgr
    #


# TODO: Draw arrows depicting the provided `flow` on a 10x10 pixel grid.
#       You may use `cv2.arrowedLine(..)`.


def drawArrows(img, flow, arrow_color = (0, 255, 0)):
    outimg = img.copy()

    point_y = []
    point_x = []

    for i in range(flow.shape[0]): # height
        for j in range(flow.shape[1]): # width
            if i % 10 == 5 and j % 10 == 5:
                point_y.append(i)
                point_x.append(j)

    for i in range(len(point_x)):
        outimg = cv2.arrowedLine(outimg, (point_x[i] , point_y[i]), (int(point_x[i] + flow[point_y[i]][point_x[i]][0]), int(point_y[i] + flow[point_y[i]][point_x[i]][1])), arrow_color, thickness=1, tipLength=1)

    return outimg



# Calculate the angular error of an estimated optical flow compared to ground truth
def calculateAngularError(estimated_flow, groundtruth_flow):
    nom = groundtruth_flow[:, :, 0] * estimated_flow[:, :, 0] + groundtruth_flow[:, :, 1] * estimated_flow[:, :, 1] + 1.0
    denom = np.sqrt((groundtruth_flow[:, :, 0] ** 2 + groundtruth_flow[:, :, 1] ** 2 + 1.0) * (estimated_flow[:, :, 0] ** 2 + estimated_flow[:, :, 1] ** 2 + 1.0))
    return (1.0 / (estimated_flow.shape[0] * estimated_flow.shape[1])) * np.sum(np.arccos(np.clip(nom / denom, 0, 1)))



# Load a flow map from a file
def load_FLO_file(filename):
    if os.path.isfile(filename) is False:
        print("file does not exist %r" % str(filename))
    flo_file = open(filename,'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    if magic != 202021.25:
        print('Magic number incorrect. .flo file is invalid')
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    # The float values for u and v are interleaved in row order, i.e., u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...,
    # In total, there are 2*w*h flow values
    data = np.fromfile(flo_file, np.float32, count=2*w[0]*h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    flo_file.close()
    # Some cleanup (remove cv-destroying large numbers)
    flow[np.sqrt(np.sum(flow ** 2, axis = 2)) > 100] = 0
    return flow
