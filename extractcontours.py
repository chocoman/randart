import numpy as np
import cv2 as cv
import os
import sys
import pdb


def remove_partial_blobs(image):
    image[0, :] = 255
    image[-1, :] = 255
    image[:, 0] = 255
    image[:, -1] = 255
    cv.imshow('border', image)
    retval, flooded, mask, rect = cv.floodFill(image, None, [0,0], 0)
    cv.imshow('border', flooded)
    return image

def vignette(image):
    height, width = image.shape
    vertical_kernel = cv.getGaussianKernel(height, 300)
    horizontal_kernel = cv.getGaussianKernel(width, 300)
    kernel_raw = vertical_kernel * horizontal_kernel.T
    kernel = kernel_raw / kernel_raw.max()
    return image * kernel

def big_blur(image):
    height, width  = image.shape
    small = cv.resize(image, None, fx = 0.05, fy = 0.05, interpolation = cv.INTER_CUBIC)
    blured = cv.GaussianBlur(small, (51, 51), 0)
    big = cv.resize(blured, (width, height), interpolation = cv.INTER_CUBIC)
    return cv.GaussianBlur(big, (51, 51), 0)

def image_to_blob(image_name):
    print(f'loading image {image_name}')
    image = cv.imread(image_name)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) / 255
    if (image.shape[0] > image.shape[1]):
        image = image.T
    image = cv.GaussianBlur(image, (171, 171), 0)
    cv.imshow('blur1', image * 255)
    blur = big_blur(image)
    image = vignette(image)
    cv.imshow('vignette', image * 255)
    image = image - blur + 0.5
    #image = cv.GaussianBlur(image, (11, 11), 0)
    ret, image = cv.threshold(image, 0.5, 255, cv.THRESH_BINARY)
    image = np.array(image * 255, np.uint8)
    cv.imshow('thresh', image * 255)
    image = remove_partial_blobs(image)
    cv.imshow('cleaned', image * 255)
    # contours, hierarchy = cv.findContours(np.array(image * 255, np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # contours_drawing = np.zeros(image.shape)
    # cv.drawContours(contours_drawing, contours, -1, (255), 3)
    # cv.imshow('contour', contours_drawing)
    return image
    '''
    for contour in contours:
        img = np.zeros(blur.shape)
        cv.drawContours(img, [contour], 0, (255), 3)
        cv.imshow('contour', img)
        ey = cv.waitKey(3000)
    '''
def visualize_blob(mask):
    image = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    color = (200, 190, 160)
    channels = []
    for channel_index in range(3):
        channel = 255 * np.ones(mask.shape, np.uint8)
        channel[mask != 0] = color[channel_index]
        channel = channel.reshape((*mask.shape, 1))
        channels.append(channel)
    image = np.concatenate(channels, axis = 2)
    cv.imshow('vis', image)
    return image



if len(sys.argv) != 2:
    print(f'usage: {sys.argv[0]} images_directory')
    sys.exit(1)
images_directory = sys.argv[1]

for file_name in os.listdir(images_directory):
    path = os.path.join(images_directory, file_name)
    blob = image_to_blob(path)
    visualize_blob(blob)
    ey = cv.waitKey(3000)

