from matplotlib.pyplot import axis
import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image
import pickle


def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1 / (1 + np.exp(-x))


def softmax(a: np.ndarray) -> np.ndarray:
  return np.exp(a) / np.sum(np.exp(a))


def get_data():
  (x_train, t_train), (x_test, t_test) = mnist.load_data()

  x_train = x_train.reshape(-1, 784)
  x_test = x_test.reshape(-1, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  x_test = x_test.astype('float32')
  x_test /= 255

  return x_test, t_test


def init_network():
  with open('/home/kohbanye/git/ORIGIN-deep-learning-from-scratch/ch03/sample_weight.pkl', 'rb') as f:
    network = pickle.load(f)

  return network


def predict(network, x):
  W1, W2, W3 = network['W1'], network['W2'], network['W3']
  b1, b2, b3 = network['b1'], network['b2'], network['b3']

  a1 = np.dot(x, W1) + b1
  z1 = sigmoid(a1)
  a2 = np.dot(z1, W2) + b2
  z2 = sigmoid(a2)
  a3 = np.dot(z2, W3) + b3

  return softmax(a3)


x, t = get_data()
network = init_network()

accuracy_count = 0
""" for i in range(len(x)):
  y = predict(network, x[i])
  p = np.argmax(y)
  if p == t[i]:
    accuracy_count += 1 """

y = predict(network, x)
p = np.argmax(y, axis=1)
accuracy_count += np.sum(p == t)

print('Accuracy: ' + str(float(accuracy_count) / len(x)))
