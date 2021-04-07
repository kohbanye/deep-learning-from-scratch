import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
  return 1 / (1 + np.exp(-x))


def gen_random_matrix(r: int, c: int) -> np.ndarray:
  l = []
  if r == 1:
    for _ in range(c):
      l.append(np.random.rand())
  else:
    for i in range(r):
      l.append([])
      for _ in range(c):
        l[i].append(np.random.rand())

  return np.array(l)


def init_network() -> dict:
  network = {}
  network['W1'] = gen_random_matrix(2, 3)
  network['b1'] = gen_random_matrix(1, 3)
  network['W2'] = gen_random_matrix(3, 2)
  network['b2'] = gen_random_matrix(1, 2)
  network['W3'] = gen_random_matrix(2, 2)
  network['b3'] = gen_random_matrix(1, 2)

  return network


def forward(network: dict, x: np.ndarray) -> np.ndarray:
  a1 = np.dot(x, network['W1']) + network['b1']
  z1 = sigmoid(a1)
  a2 = np.dot(z1, network['W2']) + network['b2']
  z2 = sigmoid(a2)
  a3 = np.dot(z2, network['W3']) + network['b3']
  
  return a3


network = init_network()
x = gen_random_matrix(1, 2)
y = forward(network, x)
print(y)
