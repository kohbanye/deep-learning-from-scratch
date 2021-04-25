import sys
import os
sys.path.append(os.pardir)
import numpy as np
from utils.functions import gradient, softmax, cross_entropy_error


class simple_net:
  def __init__(self) -> None:
    self.W = np.random.randn(2, 3)

  def predict(self, x: np.ndarray) -> np.ndarray:
    return np.dot(x, self.W)

  def loss(self, x, t):
    z = self.predict(x)
    y = softmax(z)
    return cross_entropy_error(y, t)


net = simple_net()
print(net.W)

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

def f(W):
  return net.loss(x, t)

print(gradient(f, net.W))
