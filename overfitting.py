import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# be sure those are floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# normalize :
x_train /= x_train.max()
x_test  /= x_test.max()

# reshape :
x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
x_test  = x_test.reshape(x_test.shape[0], 32, 32, 3)

# one hot encode for labels
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_test  = tf.keras.utils.to_categorical(y_test,  num_classes=10, dtype='float32')

# Model definition
# the model is impossible to decifer.
model = tf.keras.models.Sequential()
model.add(Input(shape=(32, 32, 3)))
model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu'))
model.add(Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), activation='relu'))

# Optimizer
opt = tf.keras.optimizers.SGD(lr=0.1)

# Compiling
model.compile(optimizer=opt, metrics=['accuracy',], loss='categorical_crossentropy' )

# Summary.
model.summary()
