#!/usr/bin/env python3

import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import sys

def openp(fil):
    return pickle.load(open(fil, 'rb'))


def montage(imglist, cols=None):
    """Do something like the matlab montage. Slow."""
    nr_images = imglist.shape[0]
    if cols is None:
        cols = 8
    rows = int(np.ceil(nr_images / cols))

    fig = plt.figure(1, (4., 4.))

    grid = ImageGrid(fig, 111,
                     nrows_ncols=(rows, cols),
                     axes_pad=0.1)

    for i in range(nr_images):
        grid[i].imshow(imglist[i])
        grid[i].axis('off')

    plt.show()


def main():
    argv = len(sys.argv)
    args = sys.argv

    if argv <= 1:
        print("Usage:", args[0], "'picklefile' 'offset/(negative for rng seed)'")
        exit(0)

    rng = None
    offset = 0

    if argv >= 3:
        r = int(args[2])
        if r < 0:
            rng = -r
        else:
            offset = r

    fil = args[1]
    m = openp(fil)

    print("Shape:", m.shape)

    if rng is not None:
        np.random.seed(rng)
        np.random.shuffle(m)

    nplen = min(64, m.shape[0])
    cols = int(np.ceil(np.sqrt(nplen)))

    montage(m[offset:offset+nplen], cols)

if __name__ == '__main__':
    main()