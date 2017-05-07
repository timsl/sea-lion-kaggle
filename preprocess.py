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

''' Bounding boxes '''
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

def countception_target(img, coords, img_n=0, size=512, padsize=33):
    n_x = img.shape[0] // size
    n_y = img.shape[1] // size

    x_rem = img.shape[0] % size
    y_rem = img.shape[0] % size

    boxsize = 20

    for x in range(n_x + (x_rem > 0)):
        for y in range(n_y + (y_rem > 0)):
            xmin = x*size
            ymin = y*size
            xmax = xmin+size 
            ymax = ymin+size

            new_img = img[xmin:xmax, ymin:ymax]
            target_img = np.zeros((size+padsize, size+padsize))

            for i in coords:
                if (i.x-boxsize) > xmin and (i.x+boxsize) < xmax and (i.y-boxsize) > ymin and (i.y+boxsize) < ymax:
                    target_img[(i.x-boxsize-xmin):(i.x+boxsize-xmax), (i.y-boxsize-ymin):(i.y+boxsize-ymin)] += 30

            cv2.imwrite('TrainSplit/%d_%d_%d.jpg' % (img_n, x, y), cv2.cvtColor(new_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite('TrainDottedSplit/%d_%d_%d.jpg' % (img_n, x, y), target_img)

for i in range(41, 42):
    img = mpimg.imread("Train/%d.jpg" % i)
    dotted_img = mpimg.imread("TrainDotted/%d.jpg" % i)
    print("## Extracting coordinates! ##")
    sealioncoords = extract_points(img, dotted_img, i)
    print("## Done extracting coordinates! ##")
    print()
    print("## Generating countception target images! ##")
    countception_target(img, sealioncoords, i, 256)
    print("## Done generating countception target images! ##")
    
    #img_boxes = draw_boxes(img, sealioncoords)
    #plt.imshow(img_boxes)
