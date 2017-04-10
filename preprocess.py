import numpy as np
import cv2
from collections import namedtuple

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

SeaLionCoord = namedtuple('SeaLionCoord', ['img_id', 'x', 'y', 'cls'])

colors = (
            (243,8,5),          # red
            (244,8,242),        # magenta
            (87,46,10),         # brown 
            (25,56,176),        # blue
            (38,174,21),        # green
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
    sealioncoords = [SeaLionCoord(img_id, x, y, np.argmin(np.linalg.norm(colors -
            dotted[x, y], axis = 1))) for x, y in coords]
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

for i in range(42, 43):
    img = mpimg.imread("Train/%d.jpg" % i)
    dotted_img = mpimg.imread("TrainDotted/%d.jpg" % i)
    sealioncoords = extract_points(img, dotted_img, i)
    img_boxes = draw_boxes(img, sealioncoords)
    plt.imshow(img_boxes)
    plt.show()
