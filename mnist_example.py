from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K

import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np

sns.set_style('darkgrid')

import numpy as np

class LossHistory(tf.keras.callbacks.Callback):

  def __init__(self):

    self.losses         = []
    self.val_losses     = []
    self.accuracies     = []
    self.val_accuracies = []
    self.pars           = []
    self.epochs         = []

  def on_train_begin(self, logs={}):
    pass

  def on_batch_end(self, batch, logs={}):
    pass

  def on_epoch_end(self, epoch, logs={}):
    self.losses.append(logs.get('loss'))
    self.accuracies.append(logs.get('accuracy'))
    self.val_losses.append(logs.get('val_loss'))
    self.val_accuracies.append(logs.get('val_accuracy'))
    self.pars.append(self.model.count_params())
    self.epochs.append(epoch+1)


# The known number of output classes.
num_classes = 10
batch_size  = 30

model_path = 'saved_models/mnist_doubleU_shaped.h5'

train_size = 4000
test_size  = 1000

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = np.reshape(x_train.astype('float32') / 255., (60000, -1))[:train_size]
x_test  = np.reshape(x_test.astype('float32')  / 255., (10000, -1))[:test_size]
y_train = tf.keras.utils.to_categorical(y_train, num_classes)[:train_size]
y_test  = tf.keras.utils.to_categorical(y_test,  num_classes)[:test_size]

def create_model(h_units):

  model = Sequential()
  model.add(Input(shape=(784)))
  model.add(Dense(h_units, activation='linear'))
  model.add(Dense(10, activation='linear'))

  opt = tf.keras.optimizers.SGD(learning_rate=0.1)

  model.compile(loss='mean_squared_error',
                optimizer=opt,
                metrics=['accuracy'])

  return model

def fit(x, y, x_test, y_test, model, epochs):

  callback = LossHistory()

  history = model.fit(x, y,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test),
                      shuffle=False,
                      callbacks=[callback],
                      )

  return history, callback

def train(x_train, y_train, x_test, y_test, range_units, epochs):

  callback   = LossHistory()
  num_iter = len(range_units)

  for i, h in enumerate(range_units):

    model = create_model(h)

    print('\r Iteration : {} / {}'.format(i, num_iter), end='\r', flush=True)

    hist = model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test),
                      shuffle=False,
                      verbose=0,
                      callbacks=[callback]
                      )

  return callback

def num_params(h_units):
  return (784 + 1) * h_units + (h_units + 1) * 10


call = train(x_train, y_train, x_test, y_test, np.arange(2, 150, 5), epochs=300)

dict = {'loss'         : call.losses,
        'val_loss'     : call.val_losses,
        'accuracy'     : call.accuracies,
        'val_accuracy' : call.val_accuracies,
        'params'       : call.pars,
        'epochs'       : call.epochs,
}

df = pd.DataFrame(dict)

df.to_csv('saved_models/mnist_doubleU_shaped.csv', header=True, float_format='%g', index=False)
