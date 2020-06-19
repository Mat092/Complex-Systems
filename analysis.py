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

histfile = 'saved_models/hist_cifar10_parameters.csv'
callfile = 'saved_models/call_cifar10_parameters.csv'

dfh = pd.read_csv(histfile)
dfc = pd.read_csv(callfile)


def plot_params():
  pass




















#%%
