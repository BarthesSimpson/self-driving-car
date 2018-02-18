# implements the NVIDIA end-to-end model for SDCs

import tensorflow as tf
import pandas as pd
from keras.layers import Activation, Dense, Flatten, Lambda, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam

import utils

model = Sequential()

# normalize
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(64, 64, 3)))

# convolutional layers
pool_size = (2, 2)
strides = (1, 1)

model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=strides))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=strides))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=strides))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=strides))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size, strides=strides))

model.add(Flatten())

# fully connected layers
model.add(Dense(1164))
model.add(Activation('relu'))

model.add(Dense(100))
model.add(Activation('relu'))

model.add(Dense(50))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('relu'))

model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(0.0001), loss="mse")

# training and validation
data = pd.read_csv('./data/driving_log.csv')
number_of_validation_samples = len(data) * 0.3
train_gen = utils.generate_next_batch()
validation_gen = utils.generate_next_batch()

history = model.fit_generator(train_gen,
                              samples_per_epoch=20032,
                              nb_epoch=8,
                              validation_data=validation_gen,
                              nb_val_samples=number_of_validation_samples,
                              verbose=1)

# save model
utils.save_model(model)
