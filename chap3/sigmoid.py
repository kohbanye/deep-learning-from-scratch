import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
  x: np.ndarray = np.arange(-7, 7, 0.1)
  y = sigmoid(x)
  plt.plot(x, y)
  plt.ylim(-0.1, 1.1)
  plt.show()
