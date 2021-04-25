import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

pi = np.pi

x = np.linspace(-3 * pi, 3 * pi, 256)
y = np.linspace(-3 * pi, 3 * pi, 256)

X, Y = np.meshgrid(x, y)

Z = (X + Y) * np.exp(-X * Y)

ax.plot_surface(X, Y, Z, cmap='summer')

ax.contour(X, Y, Z, colors='black', offset=-1)

plt.show()
