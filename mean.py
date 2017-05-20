#!/usr/bin/env python3

import pickle
import sys

import numpy as np


def openp(file_name):
    return pickle.load(open(file_name, 'rb'))


def pickle_save(x, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(x, f)





def main():
    PICKLE_COUNT = 9

    mean = 0
    img_count = 0

    for num in range(PICKLE_COUNT):
        c = openp("train_" + str(num) + "_c.p")
        mean += np.sum(c)
        img_count += c.shape[0]

    mean /= img_count

    t = openp("test_c.p")
    print("TESTMSE:", np.mean((t-mean)**2))


if __name__ == "__main__":
    main()
