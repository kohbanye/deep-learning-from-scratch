import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image


def img_show(img):
  pil_img = Image.fromarray(np.uint8(img))
  pil_img.show()


(x_train, t_train), (x_test, t_test) = mnist.load_data()

x_train: np.ndarray = x_train.reshape(-1,784)
x_test: np.ndarray = x_test.reshape(-1,784)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
