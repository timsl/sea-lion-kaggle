import numpy as np
import cv2
from collections import namedtuple
import pickle

import matplotlib.image as mpimg

SeaLionCoord = namedtuple('SeaLionCoord', ['img_id', 'x', 'y', 'cls'])

colors = (
            (243, 8, 5),          # red
            (244, 8, 242),        # magenta
            (87, 46, 10),         # brown
            (25, 56, 176),        # blue
            (38, 174, 21),        # green
            )

sizes = (45, 38, 30, 25, 25)

BADIMAGES = [3, 7, 9, 21, 30, 34, 71, 81, 89, 97, 151, 184, 215, 234,
             242, 268, 290, 311, 331, 344, 380, 384, 406, 421, 469, 475, 490,
             499, 507, 530, 531, 605, 607, 614, 621, 638, 644, 687, 712, 721,
             767, 779, 781, 794, 800, 811, 839, 840, 869, 882, 901, 903, 905,
             909, 913, 927, 946]


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
    sealioncoords = [SeaLionCoord(img_id, x, y, np.argmin(np.linalg.norm(colors - dotted[x, y], axis=1))) for x, y in coords]
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

def countception_target(img, coords, img_n=0, size=256, padsize=16):
    n_x = img.shape[0] // size
    n_y = img.shape[1] // size

    x_rem = img.shape[0] % size
    y_rem = img.shape[1] % size

    imgs = []
    target_imgs = []
    counts = []

    for x in range(n_x + (x_rem > 0)):
        for y in range(n_y + (y_rem > 0)):
            count = 0
            xmin = x*size
            ymin = y*size
            xmax = xmin+size
            ymax = ymin+size

            new_img = img[xmin:xmax, ymin:ymax]

            # Edge case
            if x == n_x or y == n_y:
                temp = np.zeros((256, 256, 3), dtype=np.uint8)
                temp[:new_img.shape[0], :new_img.shape[1], :] = new_img
                new_img = temp

            target_img = np.zeros((size+padsize*2, size+padsize*2), dtype=np.uint8)

            for i in coords:
                # In 256x256-coords
                unpad_x = i.x - xmin
                unpad_y = i.y - ymin
                if unpad_x < 0 or unpad_y < 0 or unpad_x >= size or unpad_y >= size:
                    continue

                # In 288x288-coords
                pad_x = unpad_x + padsize
                pad_y = unpad_y + padsize
                assert pad_x - padsize >= 0
                assert pad_x + padsize < size + 2*padsize
                assert pad_y - padsize >= 0
                assert pad_y + padsize < size + 2*padsize

                target_img[(pad_x - padsize):(pad_x + padsize), (pad_y - padsize):(pad_y + padsize)] += 1
                count += 1
            imgs.append(new_img)
            target_imgs.append(target_img)
            counts.append(count)

    return imgs, target_imgs, counts


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


def pickle_many(start, stop, file_prefix):
    imgs = []
    target_imgs = []
    counts = []

    for i in range(start, stop):
        if i in BADIMAGES:
            print("!! SKIPPED Image %d !!" % i)
            continue

        print("## Image %d ##" % i)
        img = mpimg.imread("Train/%d.jpg" % i)
        dotted_img = mpimg.imread("TrainDotted/%d.jpg" % i)
        print("## Extracting coordinates! ##")
        sealioncoords = extract_points(img, dotted_img, i)
        print("## Done extracting coordinates! ##")
        print()
        print("## Generating countception target images! ##")
        img, target_img, count = countception_target(img, sealioncoords, i, 256)
        imgs.extend(np.array(img))
        target_imgs.extend(np.array(target_img))
        counts.extend(np.array(count))
        print("## Done generating countception target images! ##")
        print()

    p_np_imgs = np.array(imgs)
    p_target_imgs = np.array(target_imgs)
    p_counts = np.array(counts)

    np_imgs, target_imgs, counts = remove_some_negative(p_np_imgs, p_target_imgs, p_counts, 0.0)

    pickle_save(np_imgs, file_prefix + "_x.p")
    pickle_save(target_imgs, file_prefix + "_y.p")
    pickle_save(counts, file_prefix + "_c.p")


for trip in [
        (0, 100, "train_0"),
        (100, 200, "train_1"),
        (200, 300, "train_2"),
        (300, 400, "train_3"),
        (400, 500, "train_4"),
        (500, 600, "train_5"),
        (600, 700, "train_6"),
        (700, 800, "train_7"),
        (800, 900, "train_8"),
        (930, 945, "test"),
        (945, 948, "valid")]:
    start, end, name = trip
    pickle_many(start, end, name)
