# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# sns.set_style('darkgrid')
#
#
# df = pd.read_csv('saved_models/monitor_mnist.csv')
#
# df.head()
#
# df['error'] = 1. - df['accuracy']
# df['test_error'] = 1. - df['val_accuracy']
#
# df.head()
#
# def plotting():
#
#   fig, ax = plt.subplots()
#
#   df.plot(x='steps', y='error', ax=ax, label='train set error', color='red', linewidth=1, marker='x')
#   df.plot(x='steps', y='test_error', ax=ax, label='test set error', color='blue', linewidth=1, marker='x')
#   fig.suptitle('Error for MNIST dataset as a function of train size');
#
#   plt.show();
#
# plotting()


#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')

dfh = pd.read_csv('saved_models/cifar100_history_conv.csv')
dfc = pd.read_csv('saved_models/cifar100_callback_conv.csv')

dfh['epoch'] = np.arange(300) + 1
dfh['error'] = 1. - dfh['accuracy']
dfh['val_error'] = 1. - dfh['val_accuracy']

dfh.head()

def plot ():

  fig, ax = plt.subplots(figsize=(10,7))

  dfh.plot(x='epoch', y='error', color='red', linewidth='1', marker='', ax=ax, label='train error')
  dfh.plot(x='epoch', y='val_error', color='blue', linewidth='1', marker='', ax=ax, label='test error')
  ax.set_title('Training over 300 epoch for CIFAR100')
  ax.set_ylabel('Classification Error')
  ax.set_ylim(0., 1.)


  plt.show()

plot()
