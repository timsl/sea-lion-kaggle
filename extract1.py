#!/usr/bin/env python3

import pickle
from collections import namedtuple

import numpy as np

import cv2
import matplotlib.image as mpimg

import matplotlib.pyplot as plt



SeaLionCoord = namedtuple('SeaLionCoord', ['img_id', 'x', 'y', 'cls'])

colors = (
            (243, 8, 5),          # red
            (244, 8, 242),        # magenta
            (87, 46, 10),         # brown
            (25, 56, 176),        # blue
            (38, 174, 21),        # green
            )

sizes = (45, 38, 30, 25, 25)

def neighbourhood(xs, ys):
    return sum((x-y)**2 for x, y in zip(xs, ys)) > 300


def remove_neighbourhood(data, cond):
    coords = []
    for element in data:
        if all(cond(element, other) for other in coords):
            coords.append(element)
    return coords


def extract_points(original, dotted, img_id):
    diff = cv2.subtract(original, dotted)
    diff[dotted < 30] = 0
    x, y = np.where(np.any(diff > 60, axis=2))
    xy = zip(x, y)
    coords = remove_neighbourhood(xy, neighbourhood)
    sealioncoords = [
        SeaLionCoord(img_id, x, y,
                     np.argmin(np.linalg.norm(colors - dotted[x, y], axis=1)))
        for x, y in coords]
    return sealioncoords


def draw_boxes(img, coords):
    xmax, ymax, _ = img.shape
    for i in coords:
        size = sizes[i.cls]
        color = colors[i.cls]
        if i.x-size > 0 and i.x+size < xmax and i.y-size > 0 and i.y+size < ymax:
            img[i.x-size:i.x+size, i.y-size] = color
            img[i.x-size:i.x+size, i.y+size] = color
            img[i.x-size, i.y-size:i.y+size] = color
            img[i.x+size, i.y-size:i.y+size] = color
    return img


def countception_target(img, coords, padsize=16):
    target_img = np.zeros((img.shape[0] + 2*padsize, img.shape[1] + 2*padsize))

    for i in coords:
        x = i.x + padsize
        y = i.y + padsize
        if x - padsize < 0 or y - padsize < 0:
            print("< 0")
        if x + padsize > img.shape[0] or y + padsize > img.shape[1]:
            print("> shape")

        target_img[(x - padsize):(x + padsize), (y - padsize):(y + padsize)] += 1

    return img, target_img, []

def remove_some_negative(x, y, c, negative_ratio=1.0):
    win_mask = c != 0
    los_mask = c == 0

    win_x = x[win_mask]
    win_y = y[win_mask]
    win_c = c[win_mask]
    los_x = x[los_mask]
    los_y = y[los_mask]
    los_c = c[los_mask]

    order = np.random.permutation(los_c.shape[0])
    amt = int(win_c.shape[0]*negative_ratio)
    los_x = los_x[order][0:amt]
    los_y = los_y[order][0:amt]
    los_c = los_c[order][0:amt]

    both_x = np.concatenate([win_x, los_x])
    both_y = np.concatenate([win_y, los_y])
    both_c = np.concatenate([win_c, los_c])
    order = np.random.permutation(both_c.shape[0])

    return both_x[order], both_y[order], both_c[order]


def pickle_save(x, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(x, f)


def load_many(start, stop):
    imgs = []
    target_imgs = []
    counts = []

    for i in range(start, stop):
        print("## Image %d ##" % i)
        img = mpimg.imread("Train/%d.jpg" % i)
        dotted_img = mpimg.imread("TrainDotted/%d.jpg" % i)
        print("## Extracting coordinates! ##")
        sealioncoords = extract_points(img, dotted_img, i)
        print("## Done extracting coordinates! ##")
        print()
        print("## Generating countception target images! ##")
        # print(sealioncoords)
        X1 = 1350
        X2 = 1900
        Y1 = 3000
        Y2 = 3350
        padsize = 16
        plt.imshow(img[X1:X2, Y1:Y2])
        plt.imsave("original.png", img[X1:X2, Y1:Y2])
        plt.show()
        img, target_img, count = countception_target(img, sealioncoords)

        print(target_img.shape)
        plt.imshow(target_img[X1+padsize:X2+padsize, Y1+padsize:Y2+padsize])
        plt.imsave("targets.png", target_img[X1+padsize:X2+padsize, Y1+padsize:Y2+padsize])
        plt.show()

        imgs.extend(np.array(img))
        target_imgs.extend(np.array(target_img))
        counts.extend(np.array(count))
        print("## Done generating countception target images! ##")
        print()

    imgs = np.array(imgs)
    target_imgs = np.array(target_imgs)
    counts = np.array(counts)

    return imgs, target_imgs, counts


def pickle_many(start, stop, negative_ratio, file_prefix):
    """Pickle images range(start,stop) into files starting with file_prefix."""
    p_np_imgs, p_target_imgs, p_counts = load_many(start, stop)


pickle_many(5, 6, 0, "ok")
