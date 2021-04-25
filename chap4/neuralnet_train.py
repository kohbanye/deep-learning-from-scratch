import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.pardir)
from utils.functions import *
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = get_data()

train_loss_list: List[int] = []

iters_num = 1000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  grad = network.numerical_gradient(x_batch, t_batch)

  for key in ('W1', 'b1', 'W2', 'b2'):
    network.params[key] -= learning_rate * grad[key]

  loss = network.loss(x_batch, t_batch)
  train_loss_list.append(loss)

x = np.linspace(0, len(train_loss_list), num=len(train_loss_list))
y = np.array(train_loss_list)

plt.plot(x, y)
plt.show()
