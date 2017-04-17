import numpy as np
import pandas as pd
import cv2

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split

n_dots_per_class = pd.read_csv('train.csv')
n_dots = n_dots_per_class.sum(numeric_only = True, axis=1) # Todo: ignore train_id



X_train = []
y_train = []

for i in range(41,51):
    img = cv2.imread("Train/%d.jpg" % i)
    if img.shape[0] != 3328:
        continue
    X_train.append(img)
    y_train.append(n_dots[i] - i) # -i because n_dots sum doesn't ignore train_id

    
X_train, X_val, y_train, y_val = train_test_split(np.array(X_train), np.array(y_train), test_size = 0.2)
    
print(X_train[0].shape)
model = Sequential()
model.add(BatchNormalization(axis=1, input_shape = (3328,4992,3)))
model.add(Convolution2D(16,3,3, border_mode='valid', subsample=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(1))
model.summary()

model.compile(loss = 'mse', optimizer= Adam(lr=0.0001))

checkpoint = ModelCheckpoint(filepath = "/models/model.h5", verbose = 1, save_best_only = True,
                             monitor = 'val_loss')

callback = EarlyStopping(monitor = 'val_loss', patience = 2, verbose = 1)

model.fit(X_train, y_train, nb_epoch = 20, verbose = 1, batch_size = 1, shuffle = True,
          validation_data= (X_val, y_val), callbacks = [checkpoint, callback])
