# An adaptation from https://github.com/ieee8023/NeuralNetwork-Examples/blob/master/theano/counting/count-ception.ipynb

import matplotlib
import matplotlib.pyplot as plt

import pickle
from keras.layers import Conv2D, BatchNormalization, Input, concatenate, ZeroPadding2D
# from keras.layers import Dense, Activation, Lambda, Conv2D, MaxPool2D, Flatten, BatchNormalization, Input, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint
import keras.models
# from keras.datasets import cifar10
import numpy as np
from skimage.io import imread  #, imsave
import scipy.misc
import sys
import glob
import os
import random
import cv2

scale = 1
patch_size = 32
framesize = 256
noutputs = 1
nsamples = 700
stride = 1
ef = (patch_size/stride)**2

imgs = []
target_imgs = []
# Training images
# Target images

np_dataset_x = np.array(pickle.load(open("data_x.p", "rb")))
np_dataset_y = np.array(pickle.load(open("data_y.p", "rb")))
np_dataset_y = np.reshape(np_dataset_y, (np_dataset_y.shape[0], np_dataset_y.shape[1], np_dataset_y.shape[2], 1))
np_dataset_c = np.array(pickle.load(open("data_c.p", "rb")))

print(np_dataset_x.shape)
print(np_dataset_y.shape)
print(np_dataset_c.shape)

n = nsamples

np_dataset_x_train = np_dataset_x[0:n]
np_dataset_y_train = np_dataset_y[0:n]
np_dataset_c_train = np_dataset_c[0:n]
print("np_dataset_x_train", len(np_dataset_x_train))

print(np.unique(np_dataset_y_train))

np_dataset_x_valid = np_dataset_x[n:2*n]
np_dataset_y_valid = np_dataset_y[n:2*n]
np_dataset_c_valid = np_dataset_c[n:2*n]
print("np_dataset_x_valid", len(np_dataset_x_valid))

np_dataset_x_test = np_dataset_x[2*n:]
np_dataset_y_test = np_dataset_y[2*n:]
np_dataset_c_test = np_dataset_c[2*n:]
print("np_dataset_x_test", len(np_dataset_x_test))

# Keras stuff

def ConvFactory(filters, kernel_size, padding, inp, name, padding_type='valid'):
    if padding != 0:
        padded = ZeroPadding2D(padding)(inp)
    else:
        padded = inp
    conv = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding_type, name=name+"_conv")(padded)
    activated = LeakyReLU(0.01)(conv)
    bn = BatchNormalization(name=name+"_bn")(activated)
    return bn

def SimpleFactory(ch_1x1, ch_3x3, inp, name):
    conv1x1 = ConvFactory(ch_1x1, 1, 0, inp, name + "_1x1")
    conv3x3 = ConvFactory(ch_3x3, 3, 1, inp, name + "_3x3")
    return concatenate([conv1x1, conv3x3])


def build_model():
    print('#'*80)
    print('# Building model...')
    print('#'*80)

    inputs = Input(shape=(256, 256, 3))
    print("inputs:", inputs.shape)
    c1 = ConvFactory(64, 3, patch_size, inputs, "c1")
    print("c1", c1.shape)
    net1 = SimpleFactory(16, 16, c1, "net1")
    print("net:", net1.shape)
    net2 = SimpleFactory(16, 32, net1, "net2")
    print("net:", net2.shape)
    net3 = ConvFactory(16, 14, 0, net2, "net3")
    print("net:", net3.shape)
    net4 = SimpleFactory(112, 48, net3, "net4")
    print("net:", net4.shape)
    net5 = SimpleFactory(64, 32, net4, "net5")
    print("net:", net5.shape)
    net6 = SimpleFactory(40, 40, net5, "net6")
    print("net:", net6.shape)
    net7 = SimpleFactory(32, 96, net6, "net7")
    print("net:", net7.shape)
    net8 = ConvFactory(32, 17, 0, net7, "net8")
    print("net:", net8.shape)
    net9 = ConvFactory(64, 1, 0, net8, "net9")
    print("net:", net9.shape)
    net10 = ConvFactory(64, 1, 0, net9, "net10")
    print("net:", net10.shape)
    # net11 = ConvFactory(1, 1, 0, net10, "net11")
    final = Conv2D(1, 1, name="final")(net10)
    print("net:", final.shape)


    model = keras.models.Model(inputs=inputs, outputs=final)
    print("Model params:", model.count_params())

    model.compile(optimizer = 'adam', loss = 'mae', metrics = ['accuracy'], learning_rate = 0.005)

    return model

def sum_count_map(m, ef=ef):
    return np.asarray([np.sum(p)/ef for p in m])

def plot_map(m, fil):
    # m is like (256, 256, 1)
    a = np.reshape(m, (m.shape[0], m.shape[1]))
    plt.imshow(a)
    plt.savefig(fil)

TRAIN=False

if TRAIN:
    batch_size = 4
    epochs = 5

    model = build_model()

    bestcheck = ModelCheckpoint(filepath="model-best.h5", verbose=1, save_weights_only=True, save_best_only=True)
    every10check = ModelCheckpoint(filepath="model-cp.{epoch:02d}-{val_loss:.2f}.h5", verbose=1, period=10, save_weights_only=True)
    hist = model.fit(np_dataset_x_train, np_dataset_y_train, epochs=epochs, batch_size = batch_size,
                     validation_data = (np_dataset_x_valid, np_dataset_y_valid), callbacks=[bestcheck, every10check])

    model.save_weights('model.h5')

else:
    model = build_model()
    model.load_weights("model.h5", by_name=True)

pred = model.predict(np_dataset_x_test, batch_size=1)
plot_map(pred[0], "ours")
plot_map(np_dataset_y_test[0], "theirs")

preds = sum_count_map(pred)
tests = np_dataset_c_test
order = np.argsort(tests)
print(preds[order])
print(tests[order])

