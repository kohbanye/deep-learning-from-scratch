import numpy as np
from tensorflow.keras.datasets import mnist


def get_data(normalize=True, one_hot_label=False):
  (x_train, t_train), (x_test, t_test) = mnist.load_data()

  x_train = x_train.reshape(-1, 784)
  x_test = x_test.reshape(-1, 784)

  if normalize:
    x_train = x_train.astype('float32')
    x_train /= 255
    x_test = x_test.astype('float32')
    x_test /= 255

  if one_hot_label:
    one_hot_train = np.array([[0 for i in range(10)] for j in range(t_train.shape[0])])
    one_hot_test = np.array([[0 for i in range(10)] for j in range(t_test.shape[0])])
    for i in range(len(t_train)):
      one_hot_train[i][t_train[i]] = 1
    for i in range(len(t_test)):
      one_hot_test[i][t_test[i]] = 1
    t_train = one_hot_train
    t_test = one_hot_test

  return (x_train, t_train), (x_test, t_test)


def cross_entropy_error(y: np.ndarray, t: np.ndarray):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  batch_size = y.shape[0]
  return -np.sum(t * np.log(y + 1e-7)) / batch_size


(x_train, t_train), (x_test, t_test) = get_data(one_hot_label=True)

print(t_train.shape)
