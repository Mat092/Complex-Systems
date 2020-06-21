from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D

import pandas as pd
import numpy as np


class LossHistory(tf.keras.callbacks.Callback):

  def on_train_begin(self, logs={}):
    self.losses = []
    self.accuracies = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))
    self.accuracies.append(logs.get('accuracy'))

  def on_epoch_end(self, epoch, logs={}):
    import pandas as pd

    df = pd.DataFrame({'accuracies' : self.accuracies,
                       'losses'     : self.losses})

    df.to_csv('saved_models/cifar10_callback_noconv.csv', header=True, float_format='%g')

batch_size  = 100
num_classes = 10

train_size = 20000 # max 50K
test_size  = 5000  # max 10K

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'saved_models/keras_cifar100_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Be sure data are float and in range [0., 1.]
x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32')  / 255.

# Input image dimensions
img_rows, img_cols, channels = 32, 32, 3

# Channels go last for TensorFlow backend
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)[:train_size]
x_test  = x_test.reshape( x_test.shape[0] , img_rows, img_cols, channels)[:test_size]
input_shape = (img_rows, img_cols, channels)

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)[:train_size]
y_test  = tf.keras.utils.to_categorical(y_test, num_classes)[:test_size]

def create_model(n_filters, input_shape, hidden_layers=2):

  model = Sequential()
  model.add(Input(shape=input_shape))

  for _ in range(hidden_layers):
    model.add(Conv2D(filters=n_filters, kernel_size=(3,3), activation='relu', padding='same'))

  model.add(Conv2D(filters=10, kernel_size=(3,3), activation='relu', padding='same'))
  model.add(GlobalAveragePooling2D(data_format='channels_last'))
  model.add(Activation(activation='softmax'))

  opt = tf.keras.optimizers.SGD(learning_rate=0.01)

  model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

  return model

def train_model_on_parameters(X, y, model, epochs, batch_size):

  callback = LossHistory()

  history = model.fit(X, y,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=False,
              callbacks=[callback])

  return history, callback

def train_model_loop (X, y, par_range=range(1, 32, 4), epochs=10, batch_size=100) :

  histories = []
  callbacks = []

  for num_filters in par_range :

    print(num_filters)

    num_params = num_parameters(num_filters)

    model = create_model(num_filters, input_shape)

    history, callback = train_model_on_parameters(X, y, model, epochs, batch_size)

    history.history['params'] = num_params

    histories.append(history.history)
    callbacks.append({'accuracy' : callback.accuracies,
                      'loss'     : callback.losses,
                      'params'   : num_params,}
                    )

  return histories, callbacks

def num_parameters (n_filt):
  return 9 * n_filt * (13 + n_filt)

histories, callbacks = train_model_loop(x_train, y_train, np.arange(1, 60, 4), epochs=20, batch_size=batch_size)

# SALVATAGGIO OF HISTORIES AND CALLLBACKS
histfile = 'saved_models/hist_cifar10_parameters'
callfile = 'saved_models/call_cifar10_parameters'

dfh = pd.DataFrame(histories)
dfc = pd.DataFrame(callbacks)

dfh.to_csv(histfile, header=True, float_format='%g', index=False)
dfc.to_csv(callfile, header=True, float_format='%g', index=False)

'''
Every list contains the specified value by number of epoch. The rows are number of parameters.
'''
