import numpy as np
from typing import List
from collections.abc import Callable
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1 / (1 + np.exp(-x))


def softmax(a: np.ndarray) -> np.ndarray:
  c = np.max(a)
  exp_a = np.exp(a - c)
  return exp_a / np.sum(exp_a)


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> int:
  d = 1e-7
  return -np.sum(t * np.log(y + d))


def gradient(f, x):
  h = 1e-4
  grad = np.zeros_like(x)

  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  print(it)
  while not it.finished:
    idx = it.multi_index
    tmp_val = x[idx]
    x[idx] = tmp_val + h
    fxh1 = f(x)
    x[idx] = tmp_val - h
    fxh2 = f(x)
    grad[idx] = (fxh1 - fxh2) / (2 * h)

    x[idx] = tmp_val
    it.iternext()

  return grad


def get_data():
  (x_train, t_train), (x_test, t_test) = mnist.load_data()

  x_train = x_train.reshape(-1, 784)
  x_test = x_test.reshape(-1, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  x_test = x_test.astype('float32')
  x_test /= 255
  t_train = to_categorical(t_train)
  t_test = to_categorical(t_test)

  return (x_train, t_train), (x_test, t_test)
