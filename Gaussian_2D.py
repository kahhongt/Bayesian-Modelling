import matplotlib
import numpy as np
import functions as fn
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def kernel(f, a, b):  # Defining the kernel / Covariance function
    alpha = 30
    if f == 'Linear':
        k = np.matmul(a, np.transpose(b))
    elif f == 'Brownian':
        k = min(a, b)
    elif f == 'Brownian Bridge':  # Not implementable yet
        k = min(a, b) - (a * b)
    elif f == 'Squared Exponential':
        k = np.exp(-alpha * np.matmul((a - b), np.transpose(a - b)))
    elif f == 'Periodic':
        k = np.exp(-np.matmul(np.sin(2 * np.pi * (a-b)), np.transpose(np.sin(2 * np.pi * (a-b)))))  # Arbitrary setting at 2pi
    else:
        k = 1  # arbitrarily set
    return k


kernel_1 = "Squared Exponential"
x = np.arange(-1, 1, 0.1)  # Set up the range of values
y = np.arange(-1, 1, 0.1)
xv, yv = np.meshgrid(x, y)
xy_comparison = np.append(fn.columnize(xv), fn.columnize(yv), axis=1)
xy_comparison_rows = xy_comparison.shape[0]

# print(xy_comparison)
# a = np.transpose(xy_comparison)[:, 10]
# b = xy_comparison[15, :]
# c = np.matmul(a, b)

C = np.zeros((xy_comparison_rows, xy_comparison_rows))  # Initialising the covariance matrix
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        C[i, j] = kernel(kernel_1, xy_comparison[i, :], xy_comparison[j, :])  # Covariance matrix


r = np.random.randn(C.shape[0], 1)  # Creates a column of random values with mean 0 and variance 1
S, V, D = np.linalg.svd(C, full_matrices=True)  # Singular Value Decomposition
diagonal_V = np.diag(V)  # Constructing a diagonal matrix
z = np.matmul(S, np.matmul(np.sqrt(diagonal_V), r))
zv = np.reshape(z, (x.size, y.size))

fig = plt.figure()
g_surface = fig.add_subplot(111, projection="3d")
# g_surface.plot_surface(xv, yv, zv, cmap=cm.coolwarm)
g_surface.scatter(xv, yv, zv, marker='.', color='black')
g_surface.set_title(kernel_1 + ' Surf Plot')
g_surface.set_xlim(-1, 1)
g_surface.set_ylim(-1, 1)
g_surface.set_zlim(-5, 5)
g_surface.set_xlabel('x axis')
g_surface.set_ylabel('y axis')
g_surface.set_zlabel('z axis')
plt.show()


print(xv)

"""
fig = plt.figure()
gaussian = fig.add_subplot(111)
for i in range(r.shape[1]):
    colour = np.random.rand(4)
    gaussian.scatter(x, z[:, i], c=colour, marker='.')  # scatter takes in arrays of the same size

gaussian.set_title(kernel_1 + ' and ' + kernel_2)
gaussian.set_xlabel('x-axis')
gaussian.set_ylabel('z-axis')
gaussian.grid(True)
plt.show()
"""


