#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.util import random_noise

from utils import *



#
# Task 1
#


# TODO: Implement the the following filter functions such that each implements the respective image filter.
#  They shall not modify the input image, but return a filtered copy.
#  Implement at least one of them without using an existing filter function, e.g. do not use the corresponding OpenCV functions `cv2._____Blur(..)`.

def filter_box(img, ksize = 5):
    # TODO: Implement the Box filter.

    # add the padding
    old_height, old_width, channels = img.shape
    new_height = old_height + 4 # for applying kernel size of 5
    new_width = old_width + 4 
    color = (0,0,0)
    copy = np.full((new_height, new_width, channels), color, dtype=np.uint8)

    # compute the center offset
    x_center = (new_width - old_width) // 2
    y_center = (new_height - old_height) // 2

    # copy img image into center of result image
    copy[y_center:y_center+old_height, 
        x_center:x_center+old_width] = img

    output = np.copy(img)
    
    kernel = np.ones((5,5))
    for k in range(channels):
        for j in range(old_width):
            for i in range(old_height):
                apply_filter = np.copy(copy[(i+2-2):(i+2+3), (j+2-2):(j+2+3), k])
                output[i][j][k] = np.sum(apply_filter * kernel) // (ksize*ksize)

    return output


def filter_sinc(img, mask_circle_diameter = .4):
    # TODO: Implement the Sinc filter using DFT. (discrete fourier transform)

    # change to gray img, do the fourier transform and shift the zero frequency to center
    copy = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dft_copy = cv2.dft(np.float32(copy),flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_copy_shift = np.fft.fftshift(dft_copy)

    # create a center mask
    rows,cols = copy.shape
    crows,ccol = rows//2,cols//2
    radius = int((cols * mask_circle_diameter) // 2)
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[(crows - radius) : (crows + radius), (ccol - radius) : (ccol + radius)] = 1

    # apply the mask and shift back the image
    f_dft_copy_shift = dft_copy_shift * mask
    f_dft_copy = np.fft.ifftshift(f_dft_copy_shift)

    # apply inverse discrete fourier transform algorithm & calculate the magnitude of real and imaginary part(intensity)
    img_back_copy = cv2.idft(f_dft_copy)
    img_back_copy = cv2.magnitude(img_back_copy[:,:,0],img_back_copy[:,:,1])

    # return the image
    return img_back_copy



def filter_gauss(img, ksize = 5):
    # TODO: Implement the Gaussian filter.
    copy = cv2.GaussianBlur(img,(ksize,ksize),0) # image, tuple of kernel size, s.d. (x,y) (if 0 then it will calculate itself and sig x y will be same)
    return copy


def filter_median(img, ksize = 5): # just take the median of the kernel, and replace the pixel with it -> good for salt & pepper
    # TODO: Implement the Median filter.
    copy =cv2.medianBlur(img, ksize)
    return copy

#
# (Task 2)
#

# def filter_XYZ(img):
#     ...


def applyFilter(filter, img):
    return globals()["filter_" + filter](img)


img1 = cv2.imread(r"C:\Users\vigne\Downloads\sheet3\sheet3\img\geometric_shapes.png")

# Simulate image noise
noise_types = ["gaussian", "poisson", "s&p"]
imgs_noise = [from0_1to0_255asUint8(random_noise(img1, mode=n)) for n in noise_types]

imgs = [("original", img1)] + [(noise + " noise", img) for noise, img in zip(noise_types, imgs_noise)]
plt.figure(figsize=(10, 3))
showImages(imgs)

# Filter noise images
filter_types = ["box", "sinc", "gauss", "median"] # , "XYZ"] # (Task 2)
imgs_noise_filtered = [(f, [(noise, applyFilter(f, img)) for noise, img in imgs]) for f in filter_types]

imgs = imgs + [(f + " filter" if noise == "original" else "", img) for f, imgs_noise in imgs_noise_filtered for noise, img in imgs_noise]
plt.figure(figsize=(15, 8))
showImages(imgs, 4, transpose = True)



#
# Task 3
#


# TODO: Simulate a picture captured in low light without noise.
#  Reduce the brightness of `img` about the provided darkening `factor`.
#  The data type of the returned image shall be the same as that of the input image.
#  Example (factor = 3): three times darker, i.e. a third of the original intensity.
def reduceBrightness(img, factor):
    copy = np.copy(img)
    copy = copy // factor
    return copy


# TODO: "Restore" the brightness of a picture captured in low light, ignoring potential noise.
#  Apply the inverse operation to `reduceBrightness(..)`.
def restoreBrightness(img, factor):
    copy = np.copy(img)
    copy = copy * factor
    return copy


img2 = cv2.imread(r"C:\Users\vigne\Downloads\sheet3\sheet3\img\couch.jpg")
imgs = [("Original", img2)]

# Reduce image brightness
darkening_factor = 3
img_dark = reduceBrightness(img2, darkening_factor)

# Restore image brightness
img_restored = restoreBrightness(img_dark, darkening_factor)

imgs = imgs + [("Low light", img_dark), ("Low light restored", img_restored)]


# Simulate multiple pictures captured in low light with noise.
num_dark_noise_imgs = 10
imgs_dark_noise = [from0_1to0_255asUint8(random_noise(img_dark, mode="poisson")) for _ in range(num_dark_noise_imgs)]


# TODO: Now try to "restore" a picture captured in low light with noise (`img_dark_noise`) using the same function as for the picture without noise.
img_dark_noise = imgs_dark_noise[0]
img_noise_restored_simple = PLACEHOLDER(img_dark_noise)
img_noise_restored_simple = restoreBrightness(img_dark_noise, darkening_factor)

imgs = imgs + [None, ("Low light with noise", img_dark_noise), ("Low light with noise restored", img_noise_restored_simple)]


# TODO: Explain with your own words why the "restored" picture shows that much noise, i.e. why the intensity of the noise in low light images is typically so high compared to the image signal.
'''
As when we take a photo in darkness, there are mainly three factors involved in the process. The Aperture, the exposure duration and
the sensitivity of the chips. As we are taking a photo in dark, if we want to collect enough photons, we need to have a proper
exposure duration and also we need to adjust the aperture properly to let enough photons pass through it. If we can't balance the things
in a good way, it will affect the depth of fields and also the chips cannot collect enough photons, that causes some noise.

And it also depends on the sensitivity of chip, if the sensitivity of the chip cannot capture the intensity of the photons, that also
causes some noise on it.
________________________________________________________________________________
'''


# TODO: Restore a picture from all the low light pictures with noise (`imgs_dark_noise`) by computing the "average image" of them.
#  Adjust the resulting brightness to the original image (using the `darkening_factor` and `num_dark_noise_imgs`).
img_noise_stack_restored = PLACEHOLDER(imgs_dark_noise[0])
img_noise_stack_restored = np.zeros(imgs_dark_noise[0].shape,np.uint8)
for img in imgs_dark_noise:
    img_noise_stack_restored = cv2.addWeighted(img_noise_stack_restored,1,img,0.1,0.0)
img_noise_stack_restored = restoreBrightness(img_noise_stack_restored,darkening_factor)


imgs = imgs + [("Low light with noise 1 ...", imgs_dark_noise[0]),
               ("... Low light with noise " + str(num_dark_noise_imgs), imgs_dark_noise[-1]),
               ("Low light stack with noise restored", img_noise_stack_restored)]
plt.figure(figsize=(15, 8))
showImages(imgs, 3)



#
# Task 4
#


def filter_sobel(img, ksize = 3):
    # TODO: Implement a sobel filter (x/horizontal + y/vertical) for the provided `img` with kernel size `ksize`.
    #  The values of the final (combined) image shall be normalized to the range [0, 1].
    #  Return the final result along with the two intermediate images.
    sobel_x = cv2.Sobel(src=img,ddepth=cv2.CV_16S,dx=1,dy=0,ksize=ksize,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
    sobel_y = cv2.Sobel(src=img,ddepth=cv2.CV_16S,dx=0,dy=1,ksize=ksize,scale=1,delta=0,borderType=cv2.BORDER_DEFAULT)
    sobel = sobel_x + sobel_y
    sobel = cv2.normalize(sobel,sobel,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)

    return sobel, sobel_x, sobel_y


def applyThreshold(img, threshold):
    # TODO: Return an image whose values are 1 where the `img` values are > `threshold` and 0 otherwise.
    copy = np.copy(img)
    for i in range(copy.shape[0]):
        for j in range(copy.shape[1]):
            if copy[i][j] > threshold:
                copy[i][j] = 1
            else:
                copy[i][j] = 0
    return copy 


def applyMask(img, mask):
    # TODO: Apply white color to the masked pixels, i.e. return an image whose values are 1 where `mask` values are 1 and unchanged otherwise.
    #  (All mask values can be assumed to be either 0 or 1)
    copy = np.copy(img)
    for i in range(copy.shape[0]):
        for j in range(copy.shape[1]):
            if mask[i][j] == 1:
                copy[i][j] = 1
    return copy
""" 
tiny remark sobel is the expression of edges between [0,1], so if the edge value is larger than thresold, we consider it as an edge and
make it as white in edge mask and we apply edge mask to the photo, if mask is white -> that pixel in the photo will be white,
otherwise the same.
"""

img3 = img2
imgs3 = [('Noise', img_noise_restored_simple),
         ('Gauss filter', filter_gauss(img_noise_restored_simple, 3)),
         ('Image stack + Gauss filter', filter_gauss(img_noise_stack_restored, 3))]

initial_threshold = .25
imgs3_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for _, img in imgs3]
imgs_sobel = [filter_sobel(img_gray) for img_gray in imgs3_gray]
imgs_thresh = [applyThreshold(img_sobel, initial_threshold) for img_sobel, _, _ in imgs_sobel]
imgs_masked = [applyMask(img3, img_thresh) for img_thresh in imgs_thresh]

def header(label, imgs, i, j = None):
    if i == 0:
        return label, (imgs[i] if j is None else imgs[i][j])
    return imgs[i] if j is None else imgs[i][j]

imgs = [[imgs3[i], header('Sobel X', imgs_sobel, i, 0),
                   header('Sobel Y', imgs_sobel, i, 1),
                   header('Sobel', imgs_sobel, i, 2),
                   header('Edge mask', imgs_thresh, i),
                   header('Stylized image', imgs_masked, i)] for i in range(len(imgs3))]
imgs = [label_and_image for img_list in imgs for label_and_image in img_list]

plt.figure(figsize=(17, 7))
plt_imgs = showImages(imgs, 6, False, padding = (.05, .15, .05, .05))

def updateImg(threshold):
    imgs_thresh = [applyThreshold(img_sobel, threshold) for img_sobel, _, _ in imgs_sobel]
    imgs_masked = [applyMask(img3, img_thresh) for img_thresh in imgs_thresh]
    imgs_masked = [convertColorImagesBGR2RGB(img_masked)[0] for img_masked in imgs_masked]
    for i in range(len(imgs3)):
        cols = len(imgs) // len(imgs3)
        plt_imgs[i * cols + 4].set_data(imgs_thresh[i])
        plt_imgs[i * cols + 5].set_data(imgs_masked[i])

ax_threshold = plt.axes([.67, .05, .27, .06])
slider_threshold = Slider(ax=ax_threshold, label='Threshold', valmin=0, valmax=1, valinit=initial_threshold, valstep=.01)
slider_threshold.on_changed(updateImg)

plt.show()