import matplotlib
import numpy as np
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def kernel(f, a, b):  # Defining the kernel / Covariance function
    alpha = 100
    if f == 'Linear':
        k = a * b
    elif f == 'Brownian':
        k = min(a, b)
    elif f == 'Brownian Bridge':
        k = min(a, b) - (a * b)
    elif f == 'Squared Exponential':
        k = np.exp(-alpha * ((a - b) ** 2))
    elif f == 'Periodic':
        k = np.exp(-(np.sin(2 * np.pi * (a-b)))**2)  # Arbitrary setting at 2pi
    else:
        k = 1  # Arbitrarily set
    return k


kernel_1 = "Brownian"
kernel_2 = "Linear"
x = np.arange(0, 1, 0.002)  # Set up the range of values
C = np.zeros((x.size, x.size))  # Initialising the covariance matrix
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        C[i, j] = kernel(kernel_1, x[i], x[j]) * kernel(kernel_2, x[i], x[j])  # Covariance matrix

r = np.random.randn(C.shape[0], 3)  # Creates a column of random values with mean 0 and variance 1
S, V, D = np.linalg.svd(C, full_matrices=True)  # Singular Value Decomposition
diagonal_V = np.diag(V)  # Constructing a diagonal matrix
z = np.matmul(S, np.matmul(np.sqrt(diagonal_V), r))

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

