from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
import pandas as pd

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

    df.to_csv('saved_models/cifar100_callback_conv.csv', header=True, float_format='%g')

batch_size = 32
num_classes = 100
epochs = 100

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'saved_models/keras_cifar100_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Input image dimensions
img_rows, img_cols = 32, 32

# Channels go last for TensorFlow backend
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test  = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test  = tf.keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Input(shape=input_shape))
# model.add(Conv2D(16, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# initiate RMSprop optimizer
opt = tf.keras.optimizers.SGD(learning_rate=0.01)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

def monitor_on_batch (x, y, x_test, y_test, model, batch_steps_size):

  x_train_r = x.reshape(-1, batch_steps_size, 28, 28, 1)
  y_train_r = y.reshape(-1, batch_steps_size, 10)

  callback = LossHistory()

  loss, test_acc  = model.evaluate(x_test, y_test, verbose=0)
  loss, train_acc = model.evaluate(x, y, verbose=0)

  accuracies = [train_acc, ]
  test_accuracies = [test_acc, ]
  batch_steps    = [0,]
  seen = 0

  for i, (X, y) in enumerate(zip(x_train_r, y_train_r)):

    history = model.fit(X, y,
                        batch_size=batch_size,
                        epochs=1,
                        validation_data=(x_test, y_test),
                        shuffle=False,
                        callbacks=[callback])

    seen += batch_steps_size

    test_accuracies.append(history.history['val_accuracy'][0])
    accuracies.append(callback.accuracies[-1])
    batch_steps.append(seen)

  return {'accuracy'     : accuracies,
          'val_accuracy' : test_accuracies,
          'steps'        : batch_steps }


# monitor = monitor_on_batch(x_train, y_train, x_test, y_test, model, 3000)

def overfit(x, y, x_test, y_test, model, epochs):

  callback = LossHistory()

  history = model.fit(x, y,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_test, y_test),
                      shuffle=False,
                      callbacks=[callback],
                      )

  return history, callback


hist, callback = overfit(x_train, y_train, x_test, y_test, model, 300)

df_history  = pd.DataFrame(hist.history)
df_callback = pd.DataFrame({'accuracies' : callback.accuracies,
                            'losses'     : callback.losses})

df_history.to_csv('saved_models/cifar100_history_noconv.csv', header=True, float_format='%g')
df_callback.to_csv('saved_models/cifar100_callback_noconv.csv', header=True, float_format='%g')

model.save(model_name)

print('Saved trained model at %s ' % model_name)
