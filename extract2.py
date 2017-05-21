#! /usr/bin/env python

import sys
import numpy as np

from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
from numba import *

from plog import *
plog_color(True)
from tqdm import *

# @jit
def mask_points(original, dotted, threshold=25):
    plog("Masking dots...")
    diff = cv2.subtract(original, dotted)
    diff[dotted <= threshold] = 0
    diff[np.where(np.any(diff > threshold, axis=2))] = np.r_[255, 255, 255]
    dotted[diff < 100] = 0
    return dotted

# @jit
def extract_points(img, threshold=75):
    plog("Extracting points...")
    colors = [["blue",  np.r_[15,  60, 170]],
              ["brown", np.r_[80,  60,  20]],
              ["red",   np.r_[245, 10,  10]],
              ["pink",  np.r_[240, 10, 240]],
              ["green", np.r_[30, 175,  10]]]
    out = dict()
    for c, v in colors:
        out[c] = np.r_[np.where(np.linalg.norm(img - v, axis=2) < threshold)]

    return out["blue"], out["brown"], out["red"], out["pink"], out["green"]
# @jit
def main():
    i = 5
    if 0:
        masked = np.load("./masked.npy")
        bl = np.load("./blue.npy")
        br = np.load("./brown.npy")
        re = np.load("./red.npy")
        pi = np.load("./pink.npy")
        gr = np.load("./green.npy")
    else:
        img = mpimg.imread("Train/%d.jpg" % i)
        img_dot = mpimg.imread("TrainDotted/%d.jpg" % i)
        masked = mask_points(img, img_dot)
        X1 = 1350
        X2 = 1900
        Y1 = 3000
        Y2 = 3350
        plt.imshow(img[X1:X2, Y1:Y2])
        plt.imsave("extracted_dots.png", masked[X1:X2, Y1:Y2])
        plt.imsave("dotted.png", img_dot[X1:X2, Y1:Y2])
        # plt.imshow(masked)
        plt.show()
        # np.save("./masked.npy", masked)
        # bl, br, re, pi, gr = extract_points(masked)
        # np.save("./blue.npy",   bl)
        # np.save("./brown.npy",  br)
        # np.save("./red.npy",    re)
        # np.save("./pink.npy",   pi)
        # np.save("./green.npy",  gr)

    # plog("blue", bl.shape)
    # plog("brown", br.shape)
    # plog("red", re.shape)
    # plog("pink", pi.shape)
    # plog("green", gr.shape)

if __name__ == "__main__":
    main()
