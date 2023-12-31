#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#
# Task 4
#

import numpy as np
import cv2
import matplotlib.pyplot as plt

def showImage(img, show_window_now = True):
    # TODO: Convert the channel order of an image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt_img = plt.imshow(img)
    if show_window_now:
        plt.show()
    return plt_img

# TODO: Load the image "img/hummingbird_from_pixabay.png" with OpenCV (`cv2`) to the variable `img` and show it with `showImage(img)`.
img = cv2.imread("C:\Vicky\Tu Braunschweig\Semester 2\Computer vision\Exercise\Exercise_02\sheet2\img\hummingbird_from_pixabay.png") # a numpy ndarray
showImage(img=img,show_window_now=True)

def imageStats(img):
    print("Image stats:")
    width = img.shape[1]
    height = img.shape[0]
    num_channel = img.shape[2]
    print(f"width: {width}")
    print(f"height: {height}")
    print(f"number of channels: {num_channel}")

# TODO: Print image stats of the hummingbird image.
imageStats(img=img)

# TODO: Change the color of the hummingbird to blue by swapping red and blue image channels.
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# TODO: Store the modified image as "blue_hummingbird.png" to your hard drive.
cv2.imwrite('blue_hummingbird.png',img)



#
# Task 5
#

from matplotlib.widgets import Slider

# Prepare to show the original image and keep a reference so that we can update the image plot later.
plt.figure(figsize=(4, 6))
plt_img = showImage(img, False)

# TODO: Convert the original image to HSV color space.
img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

def img_update(hue_offset):
    print("Set hue offset to " + str(hue_offset))
    # TODO: Change the hue channel of the HSV image by `hue_offset`.
    # Mind that hue values in OpenCV range from 0-179.
    h = img[:,:,0] # 360 value for color
    s = img[:,:,1] # 0 to 100 for grey value
    v = img[:,:,2] # 0 to 100 for the bright
    h_new = cv2.add(h, hue_offset)
    hsv = cv2.merge([h_new,s,v])

    # TODO: Convert the modified HSV image back to RGB
    # and update the image in the plot window using `plt_img.set_data(img_rgb)`.
    img_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    plt_img.set_data(img_rgb)


# Create an interactive slider for the hue value offset.
ax_hue = plt.axes([0.1, 0.04, 0.8, 0.06]) # x, y, width, height
slider_hue = Slider(ax=ax_hue, label='Hue', valmin=0, valmax=180, valinit=0, valstep=1)
slider_hue.on_changed(img_update)

# Now actually show the plot window
plt.show()
