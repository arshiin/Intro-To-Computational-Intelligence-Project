import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(0, 15, (501, 3))
n = 0.001  # learning rate
w = n * np.random.rand(3, 3)
y = np.zeros((501, 3))
mse = np.zeros((500, 1))
x[:, 2] = 0  # 2D learning plane in a 3D plot

for i in range(0, 500):
    y[i:i + 1] = np.dot(x[i:i + 1], w)
    e = x[i:i + 1] - y[i:i + 1]
    w = w + np.dot(n * x[i:i + 1].transpose(), e)
    mse[i:i + 1] = np.square(e).mean() # mean square error

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(projection='3d')
ax.scatter3D(x[0:500, 0], x[0:500, 1], x[0:500, 2], marker='.', c='blue')
ax.scatter3D(y[0:500, 0], y[0:500, 1], y[0:500, 2], marker='*', c='hotpink')

x[500:501] = [4, 6, 10]  # input which is out of the 2D learning plane
y[500:501] = np.dot(x[500:501], w)
ax.scatter3D(x[500:501, 0], x[500:501, 1], x[500:501, 2], marker='v', c='red')
ax.scatter3D(y[500:501, 0], y[500:501, 1], y[500:501, 2], marker='s', c='red')
plt.show()

t = np.linspace(0, 500, 500)
plt.plot(t, mse)
plt.show()
