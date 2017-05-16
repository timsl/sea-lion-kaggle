#!/usr/bin/env python3

import pickle
import sys

import numpy as np

import keras.models
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.layers import (BatchNormalization, Conv2D, Input, ZeroPadding2D,
                          concatenate)
from keras.layers.advanced_activations import LeakyReLU

# Whatever was used in preprocessing
PATCH_SIZE = 32


def openp(file_name):
    return pickle.load(open(file_name, 'rb'))


def pickle_save(x, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(x, f)


def load_triple(fil_prefix):
    x = openp(fil_prefix + "_x.p")
    y = openp(fil_prefix + "_y.p")
    y = np.reshape(y, (y.shape[0], y.shape[1], y.shape[2], 1))
    c = openp(fil_prefix + "_c.p")

    return x, y, c


def train_generator(batch_size):
    NR_PICKLES = 9

    while 1:
        for p in range(NR_PICKLES):
            x, y, c = load_triple("train_" + str(p))
            for i in range(x.shape[0]//batch_size):  # Skip odds
                yield (x[i*batch_size:(i+1)*batch_size],
                       y[i*batch_size:(i+1)*batch_size])


# Keras stuff

def ConvFactory(filters, kernel_size, padding, inp, name, padding_type='valid'):
    if padding != 0:
        padded = ZeroPadding2D(padding, name=name+"_pad")(inp)
    else:
        padded = inp
    conv = Conv2D(filters=filters, kernel_size=kernel_size,
                  padding=padding_type, name=name+"_conv")(padded)
    activated = LeakyReLU(0.01, name=name+"_relu")(conv)
    bn = BatchNormalization(name=name+"_bn")(activated)
    return bn


def Inception(ch_1x1, ch_3x3, inp, name):
    conv1x1 = ConvFactory(ch_1x1, 1, 0, inp, name + "_1x1")
    conv3x3 = ConvFactory(ch_3x3, 3, 1, inp, name + "_3x3")
    return concatenate([conv1x1, conv3x3], name=name)


def Inception3(ch_1x1, ch_3x3, ch_5x5, inp, name):
    conv1x1 = ConvFactory(ch_1x1, 1, 0, inp, name + "_1x1")
    conv3x3 = ConvFactory(ch_3x3, 3, 1, inp, name + "_3x3")
    conv5x5 = ConvFactory(ch_5x5, 5, 2, inp, name + "_5x5")
    return concatenate([conv1x1, conv3x3, conv5x5], name=name)


def build_model():
    print('#'*80)
    print('# Building model...')
    print('#'*80)

    inputs = Input(shape=(256, 256, 3))
    c1 = ConvFactory(64, 3, PATCH_SIZE, inputs, "c1")
    net1 = Inception3(24, 20, 10, c1, "net1")
    net2 = Inception3(24, 20, 10, net1, "net2")
    red1 = ConvFactory(32, 1, 0, net2, "red1")
    net3 = ConvFactory(32, 15, 0, red1, "net3")
    net4 = Inception3(112, 40, 20, net3, "net4")
    net5 = Inception3(48, 40, 20, net4, "net5")
    net6 = Inception3(48, 40, 20, net5, "net6")
    net7 = Inception3(64, 48, 24, net6, "net7")
    red2 = ConvFactory(32, 1, 0, net7, "red2")
    net8 = ConvFactory(32, 17, 0, red2, "net8")
    net9 = ConvFactory(64, 1, 0, net8, "net9")
    net10 = ConvFactory(64, 1, 0, net9, "net10")
    final = Conv2D(1, 1, name="final")(net10)
    final_relu = LeakyReLU(0.01, name="final_relu")(final)

    model = keras.models.Model(inputs=inputs, outputs=final_relu)
    model.summary()

    model.compile(optimizer='adam', loss='mse', learning_rate=0.0001)

    return model


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        plt.plot(self.losses)
        plt.plot(self.val_losses)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.savefig("loss_plot")
        plt.clf()

        train_val_loss = np.column_stack((np.asarray(self.losses), np.asarray(self.val_losses)))
        np.savetxt("loss.csv", train_val_loss, delimiter=",")


def sum_count_map(m, ef=PATCH_SIZE):
    return np.asarray([np.sum(p)/ef**2 for p in m])


def plot_map(m, fil):
    # m is like (256, 256, 1)
    a = np.reshape(m, (m.shape[0], m.shape[1]))
    plt.imshow(a)
    plt.savefig(fil)


TRAIN = 1
SAVE_PICKLE = True
SAVE_ONE = False
PRINT_COUNTS = False
PRINT_MSE = True

if len(sys.argv) >= 2:          # Have some command
    TRAIN = 0
    SAVE_PICKLE = 0
    SAVE_ONE = 0
    PRINT_COUNTS = 0
    PRINT_MSE = 0

    cmd = sys.argv[1]
    if cmd == 'train':
        TRAIN = 1
    elif cmd == 'test':
        TRAIN = 0
        SAVE_PICKLE = 1
        PRINT_MSE = 1
    elif cmd == 'print_model':
        build_model()
        exit(0)
    else:
        print("Command '", cmd, "' not recognized.")
        exit(1)

np_dataset_x_valid, np_dataset_y_valid, np_dataset_c_valid = load_triple("valid")
np_dataset_x_test, np_dataset_y_test, np_dataset_c_test = load_triple("test")


if TRAIN:
    batch_size = 4
    epochs = 50

    model = build_model()

    history = LossHistory()
    saver = ModelCheckpoint(filepath="model-cp.{epoch:02d}-{val_loss:.2f}.h5",
                            verbose=1, save_weights_only=True)
    bestsaver = ModelCheckpoint(filepath="model.h5", save_best_only=True,
                                verbose=1, save_weights_only=True)

    hist = model.fit_generator(train_generator(batch_size), epochs=epochs,
                               validation_data=(np_dataset_x_valid,
                                                np_dataset_y_valid),
                               steps_per_epoch=500,
                               callbacks=[bestsaver, saver, history])

else:
    model = build_model()
    model.load_weights("model.h5", by_name=True)

if SAVE_PICKLE or SAVE_ONE or PRINT_COUNTS or PRINT_MSE:
    pred = model.predict(np_dataset_x_test, batch_size=1)
    pred_count = sum_count_map(pred)

if SAVE_PICKLE:
    pickle_save(pred, "our_test.p")

if SAVE_ONE:
    plt.imshow(np_dataset_x_test[-1])
    plt.savefig("orig")
    plot_map(pred[-1], "ours")
    plot_map(np_dataset_y_test[-1], "theirs")

if PRINT_COUNTS:
    order = np.argsort(np_dataset_c_test)
    print(pred_count[order])
    print(np_dataset_c_test[order])

if PRINT_MSE:
    print('!'*40)
    print("Test MSE:", np.mean((pred_count-np_dataset_c_test)**2))
    print('!'*40)
