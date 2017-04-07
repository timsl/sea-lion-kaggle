import numpy as np
import cv2
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SeaLionCoord = namedtuple('SeaLionCoord', ['img_id', 'x', 'y', 'cls'])

def neighbourhood(xs, ys):
    return sum((x-y)**2 for x, y in zip(xs, ys)) > 100

def remove_neighbourhood(data, cond):
    coords = []
    for element in data:
        if all(cond(element, other) for other in coords):
            coords.append(element)
    return coords

def extract_points(original, dotted, img_id):
    diff = cv2.subtract(original, dotted)
    diff[dotted < 20] = 0
    x, y = np.where(np.any(diff > 30, axis=2))
    xy = zip(x, y)
    coords = remove_neighbourhood(xy, neighbourhood)
    sealioncoords = [SeaLionCoord(img_id, x, y, dotted[x, y]) for x, y in coords]
    return sealioncoords

def draw_boxes(img, coords):
    for i in coords:
        img[i.x-10:i.x+10, i.y-10] = i.cls
        img[i.x-10:i.x+10, i.y+10] = i.cls
        img[i.x-10, i.y-10:i.y+10] = i.cls
        img[i.x+10, i.y-10:i.y+10] = i.cls
    plt.imshow(img)
    plt.show()

for i in range(41, 42):
    img = mpimg.imread("Train/%d.jpg" % i)
    dotted_img = mpimg.imread("TrainDotted/%d.jpg" % i)
    sealioncoords = extract_points(img, dotted_img, i)
    draw_boxes(img, sealioncoords)
