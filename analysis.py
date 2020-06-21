#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%

sns.set_style('darkgrid')

dfh = pd.read_csv('saved_models/cifar10_history_conv.csv')
dfc = pd.read_csv('saved_models/cifar10_callback_conv.csv')

dfh['epoch'] = np.arange(300) + 1
dfh['error'] = 1. - dfh['accuracy']
dfh['val_error'] = 1. - dfh['val_accuracy']

def plot_error_loss():

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7))

  dfh[:100].plot(x='epoch', y='loss', color='red', linewidth='1', marker='+', ax=ax, label='train loss')
  dfh[:100].plot(x='epoch', y='val_loss', color='blue', linewidth='1', marker='+', ax=ax, label='test loss')
  ax.set_title('Loss over 100 epoch for CIFAR-10', fontsize=20)
  ax.set_ylabel('Categorical CE loss', fontsize=15)
  ax.set_xlabel('Epochs', fontsize=15)
  ax.legend(fontsize=15)

  # plt.savefig('images/cifar10_100epochs_loss.png')

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,7))

  dfh[:100].plot(x='epoch', y='error', color='red', linewidth='1', marker='+', ax=ax, label='train error')
  dfh[:100].plot(x='epoch', y='val_error', color='blue', linewidth='1', marker='+', ax=ax, label='test error')
  ax.set_title('Error over 100 epoch for CIFAR-10', fontsize=20)
  ax.set_ylabel('Classification Error', fontsize=15)
  ax.set_xlabel('Epochs', fontsize=15)
  ax.legend(fontsize=15)
  # ax.set_ylim(0., 1.)

  # plt.savefig('images/cifar10_100epochs_error.png')
  plt.show();

plot_error_loss()

#%% Parameters study

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

histfile = 'saved_models/hist_cifar10_parameters'
callfile = 'saved_models/call_cifar10_parameters'

dfh = pd.read_csv(histfile)
dfc = pd.read_csv(callfile)

names      = dfh.columns.drop('params')
epochs_num = len(eval(dfh['loss'][0]))
num_list   = len(dfh['loss'])

epochs = np.asarray([ [i for i in range(epochs_num)] * num_list]).ravel() + 1
params = np.asarray([ [par] * epochs_num for par in dfh['params'] ] ).ravel()
dict   = {name : np.concatenate([ eval(lst) for lst in dfh[name]]) for name in names}

hist_data = pd.DataFrame(dict)
hist_data['params'] = params
hist_data['epoch']  = epochs


def plot_params():

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

  epoch_cond = hist_data['epoch'] == 20

  hist_data[epoch_cond].plot(y='val_loss', x='params', ax=ax)

  plt.show();


plot_params()

#%% Analysis mnist data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

callfile = 'saved_models/mnist_doubleU_shaped.csv'

df = pd.read_csv(callfile)

df['error']     = 1 - df['accuracy']
df['val_error'] = 1-df['val_accuracy']

# Data extraction
epoch_cond = df['epochs'] == np.unique(df['epochs'].unique()).max()

data = df[epoch_cond]

def plot():

  fig, (ax, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

  data.plot(ax=ax, x='params', y='loss', label='train_loss', marker='+')
  data.plot(ax=ax, x='params', y='val_loss', label='test_loss', marker='+')
  ax.legend(fontsize=15)

  data.plot(ax=ax1, x='params', y='error', label='train_error', marker='+')
  data.plot(ax=ax1, x='params', y='val_error', label='test_error', marker='+')
  ax1.legend(fontsize=15)

  plt.show();

plot()













#%%
