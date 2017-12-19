import pandas as pd
import math
import matplotlib
import numpy as np
import time
import functions as fn
import scipy
import scipy.special as scispec
import scipy.optimize as scopt

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Preamble on LGCP
1. Used to infer non-stationary Poisson Processes. Need to guarantee positive definite covariance matrices.
2, A Cox Process is a non-homogeneous Poisson Point Process where 
the underlying intensity landa is itself a stochastic process
3. We assume that the log-intensity of the point process is a GP, and we have to take the exponential of the gaussian 
surface to obtain the latent intensity surface

We need to optimise by trying to maximise the posterior, which will give us the greatest possibility of
the latent intensity, given the data. Our goal is to predict log(landa) = v, not landa directly

LGCPs are not capable of producing predictions for any input vector, but only for inputs 
corresponding to the observations.

The hessian is the the second derivative of the log-likelihood function

Proper Steps to follow:
1. Assume arbitrary covariance matrix parameters for the base case  - theta is fixed at an optimal value
2. Assume p(theta) is a delta function at those parameters
3. Goal here is to obtain v = log(landa), where landa is the actual intensity.
4. We are here predicting the posterior for v instead of landa
5. v follows a gaussian process, and there is only one optimal value set of values
6. m refers to the number of bins/quadrats

Tabulating Posterior
a. Numerator:
    - Product of poisson distributions across the data set and gaussian process prior on v
    - This is tabulated at an arbitrary set of hyperparameters - to be optimised later
b. Denominator: Intractable integral
    - 

7. Tabulate optimal v values using laplace approximation for the denominator
8. Tabulate optimal covariance from the Hessian
"""


# Define Poisson Distribution function for each quadrat - homogeneous Poisson
def poisson_discrete(k, landa):  # Takes in two parameters intensity landa and observation value k
    numerator = np.power(landa, k) * np.exp(-1 * landa)  # mean and covariance are both landa
    denominator = math.factorial(k)
    p = numerator / denominator
    return p


def poisson_cont(k, landa):  # to allow for non-integer k values
    numerator = np.power(landa, k) * np.exp(-1 * landa)
    denominator = scispec.gamma(k)  # Generalised factorial function for non-integer k values
    p = numerator / denominator
    return p


# Create log likelihood for all quadrats, for both homogeneous and non-homogeneous PPP
def poisson_product(k_array, landa_array):
    """Takes in 2 arrays of equal size, and takes product of poisson distributions"""
    quadrats = len(k_array)  # define the number of quadrats in total
    prob_array = np.zeros(quadrats)

    if np.array(landa_array.shape).size == 1:
        for i in range(len(k_array)):
            prob_array[i] = poisson_cont(k_array[i], landa_array)
            p_likelihood = np.prod(prob_array)
        else:
            if len(k_array) == len(landa_array):
                for i in range(len(prob_array)):
                    prob_array[i] = poisson_cont(k_array[i], landa_array[i])
                p_likelihood = np.prod(prob_array)
            else:
                print('Length Mismatch')
    # Note output is a scalar (singular value)
    return p_likelihood  # Returns the non logarithmic version.


def mean_func_zero(c):  # Prior mean function taken as 0 for the entire sampling range
    if np.array([c.shape]).size == 1:
        mean_c = np.ones(1) * 0  # Make sure this is an array
    else:
        mean_c = np.ones(c.shape[1]) * 0
    return mean_c  # Outputs a x and y coordinates, created from the mesh grid


def mean_func_scalar(mean, c):  # Assume that the prior mean is a constant to be optimised
    if np.array([c.shape]).size == 1:
        mean_c = np.ones(1) * mean
    else:
        mean_c = np.ones(c.shape[1]) * mean
    return mean_c


def squared_exp_2d(sigma_exp, length_exp, x1, x2):  # Only for 2-D
    # Define horizontal and vertical dimensions of covariance matrix c
    if np.array([x1.shape]).size == 1 and np.array([x2.shape]).size != 1 and x1.size == x2.shape[0]:
        rows = 1
        columns = x2.shape[1]
    elif np.array([x2.shape]).size == 1 and np.array([x1.shape]).size != 1 and x2.size == x1.shape[0]:
        rows = x1.shape[1]
        columns = 1
    elif np.array([x1.shape]).size == 1 and np.array([x2.shape]).size == 1 and x1.size == x2.size:
        rows = 1
        columns = 1
    else:
        rows = x1.shape[1]
        columns = x2.shape[1]

    c = np.zeros((rows, columns))

    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if np.array([x1.shape]).size == 1 and np.array([x2.shape]).size != 1:
                diff = x1 - x2[:, j]
            elif np.array([x1.shape]).size != 1 and np.array([x2.shape]).size == 1:
                diff = x1[:, i] - x2
            elif np.array([x1.shape]).size == 1 and np.array([x2.shape]).size == 1:
                diff = x1 - x2
            else:
                diff = x1[:, i] - x2[:, j]

            euclidean = np.sqrt(np.matmul(diff, np.transpose(diff)))
            exp_power = np.exp(-1 * (euclidean ** 2) * (length_exp ** -2))
            c[i, j] = (sigma_exp ** 2) * exp_power

    return c  # Note that this creates the covariance matrix directly


def matern_2d(v_value, sigma_matern, length_matern, x1, x2):  # there are only two variables in the matern function
    # Define horizontal and vertical dimensions of covariance matrix c
    if np.array([x1.shape]).size == 1 and np.array([x2.shape]).size != 1 and x1.size == x2.shape[0]:
        rows = 1
        columns = x2.shape[1]
    elif np.array([x2.shape]).size == 1 and np.array([x1.shape]).size != 1 and x2.size == x1.shape[0]:
        rows = x1.shape[1]
        columns = 1
    elif np.array([x1.shape]).size == 1 and np.array([x2.shape]).size == 1 and x1.size == x2.size:
        rows = 1
        columns = 1
    else:
        rows = x1.shape[1]
        columns = x2.shape[1]

    c = np.zeros((rows, columns))

    if v_value == 1/2:
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                if np.array([x1.shape]).size == 1 and np.array([x2.shape]).size != 1:
                    diff = x1 - x2[:, j]
                elif np.array([x1.shape]).size != 1 and np.array([x2.shape]).size == 1:
                    diff = x1[:, i] - x2
                elif np.array([x1.shape]).size == 1 and np.array([x2.shape]).size == 1:
                    diff = x1 - x2
                else:
                    diff = x1[:, i] - x2[:, j]

                euclidean = np.sqrt(np.matmul(diff, np.transpose(diff)))
                exp_term = np.exp(-1 * euclidean * (length_matern ** -1))
                c[i, j] = (sigma_matern ** 2) * exp_term

    if v_value == 3/2:
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                if np.array([x1.shape]).size == 1 and np.array([x2.shape]).size != 1:
                    diff = x1 - x2[:, j]
                elif np.array([x1.shape]).size != 1 and np.array([x2.shape]).size == 1:
                    diff = x1[:, i] - x2
                elif np.array([x1.shape]).size == 1 and np.array([x2.shape]).size == 1:
                    diff = x1 - x2
                else:
                    diff = x1[:, i] - x2[:, j]

                euclidean = np.sqrt(np.matmul(diff, np.transpose(diff)))
                coefficient_term = (1 + np.sqrt(3) * euclidean * (length_matern ** -1))
                exp_term = np.exp(-1 * np.sqrt(3) * euclidean * (length_matern ** -1))
                c[i, j] = (sigma_matern ** 2) * coefficient_term * exp_term
    return c
# Both kernel functions take in numpy arrays of one row (create a single column first)


def mu_post(p_mean, xy_next, c_auto, c_cross, mismatch):  # Posterior mean
    if c_cross.shape[1] != c_auto.shape[1]:
        print('First Dimension Mismatch!')
    if c_auto.shape[0] != (np.transpose(mismatch)).shape[0]:
        print('Second Dimension Mismatch!')
    else:
        mean_post = mean_func_scalar(p_mean, xy_next) + \
                    fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(mismatch))
        return mean_post


def cov_post(c_next_auto, c_cross, c_auto):  # Posterior Covariance
    c_post = c_next_auto - fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(c_cross))
    return c_post


def log_model_evidence(param, *args):  # Param includes both sigma and l, arg is passed as a pointer
    sigma = param[0]  # param is a tuple containing 2 things, which has already been defined in the function def
    length = param[1]
    noise = param[2]  # Over here we have defined each parameter in the tuple, include noise
    scalar_mean = param[3]
    xy_coordinates = args[0]  # This argument is a constant passed into the function
    histogram_data = args[1]  # Have to enter histogram data as well
    matern_nu = args[2]  # Arbitrarily chosen v value
    prior_mu = mean_func_scalar(scalar_mean, xy_coordinates)  # This creates a matrix with 2 rows
    c_auto = matern_2d(matern_nu, sigma, length, xy_coordinates, xy_coordinates)
    # c_auto = squared_exp_2d(sigma, length, xy_coordinates, xy_coordinates)
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    c_auto_noise = c_auto + c_noise  # Overall including noise, plus include any other combination
    model_fit = - 0.5 * fn.matmulmul(histogram_data - prior_mu, np.linalg.inv(c_auto_noise),
                                     np.transpose(histogram_data - prior_mu))
    model_complexity = - 0.5 * math.log(np.linalg.det(c_auto_noise))
    model_constant = - 0.5 * len(histogram_data) * math.log(2*np.pi)
    log_model_evid = model_fit + model_complexity + model_constant
    return -log_model_evid  # We want to maximize the log-likelihood, meaning the min of negative log-likelihood


"""Collate Data Points from PP_Data"""
time_start = time.clock()  # Start computation time measurement

"""Extract Data from csv"""  # Arbitrary Point Process Data
A = np.genfromtxt('PP_Data_2D.csv', delimiter=',')  # Extract from csv using numpy
df = pd.read_csv('PP_Data_2D.csv')  # Generates a DataFrame from csv - coal data
x = np.ravel(df.values[0])
y = np.ravel(df.values[1])

"""Specify rotation matrix for data set"""
theta = 0 * np.pi  # Specify degree of rotation in a clockwise direction in radians
mat_transform = np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
df_matrix = np.vstack((x, y))
df_transform = mat_transform * df_matrix
x_transform = np.ravel(df_transform[0])
y_transform = np.ravel(df_transform[1])

"""Bin point process data"""
histo, x_edges, y_edges = np.histogram2d(x_transform, y_transform, bins=10)
xv_trans_data, yv_trans_data = np.meshgrid(x_edges, y_edges)
xv_trans_data = xv_trans_data[:-1, :-1]  # Removing the last bin edge and zero points to make dimensions consistent
yv_trans_data = yv_trans_data[:-1, :-1]  # Contains a square matrix
xv_trans_row = fn.row_create(xv_trans_data)  # Creates a row from the square matrix
yv_trans_row = fn.row_create(yv_trans_data)
histo = fn.row_create(histo)
# xv_transform_row = xv_transform_row[histo != 0]  # Remove data point at histogram equal 0
# yv_transform_row = yv_transform_row[histo != 0]
# histo = histo[histo != 0]  # This is after putting them into rows
xy_data_coord = np.vstack((xv_trans_row, yv_trans_row))  # location of all the data points

"""
Data point coordinates are now at bottom-left hand corner, coordinates of data points have
to be centralised to the centre of each quadrat
"""


print(xy_data_coord.shape)

"""Note the above relates to obtaining the data set first"""
"""Generate Gaussian Surface which forms the basis of the latent intensity function"""

# Select Optimization method for hyper-parameters and other conditions
opt_method = 'nelder-mead'
matern_v = 3/2  # Define matern_v
xyz_data = (xy_data_coord, histo, matern_v)
boundary_self = [(0, 30), (0, 3), (0, 3), (0, 10)]  # Can even use this into Nelder-Mead

"""Optimization using self-made function"""
param_optimal = fn.optimise_param(opt_func=log_model_evidence, opt_arg=xyz_data, opt_method=opt_method,
                                  boundary=boundary_self)

sigma_optimal = param_optimal[0]
length_optimal = param_optimal[1]
noise_optimal = param_optimal[2]
mean_optimal = param_optimal[3]  # Getting the optimal parameters


"""Generate Covariance Matrix from Sampling Points"""
C_auto = matern_2d(matern_v, sigma_optimal, length_optimal, xy_data_coord, xy_data_coord)
C_auto = C_auto + np.eye(C_auto.shape[0]) * noise_optimal  # include noise


"""Generate Gaussian Random field - but just one example"""
r = np.random.randn(C_auto.shape[0], 1)  # Creates a column of random values with mean 0 and variance 1
S, V, D = np.linalg.svd(C_auto, full_matrices=True)  # Singular Value Decomposition
diagonal_V = np.diag(V)  # Constructing a diagonal matrix from the decomposed matrix
z = fn.matmulmul(S, np.sqrt(diagonal_V), r)  # This is a column containing the gaussian random surface values
zv = np.reshape(z, (x_edges[:-1].size, y_edges[:-1].size))  # Reshape to 2-D matrix form - bin in the same way as data
# Generated a Gaussian Random Field - Note that this gaussian random surface keeps changing


"""Measure Marginal Log-likelihood for data points"""
data_exp = np.exp(histo)  # This gives the k-array
# likelihood_total = poisson_likelihood(data_exp, z_exp)
# log_likelihood_total = np.log(likelihood_total)

lgcp = plt.figure()
g_surface = lgcp.add_subplot(221, projection="3d")
g_surface.plot_surface(xv_trans_data, yv_trans_data, zv, cmap='RdBu')
g_surface.scatter(xv_trans_row, yv_trans_row, z, marker='.', color='black')
g_surface.set_title('Gaussian Random Field (Matern v = 3/2)')
g_surface.set_xlabel('x-axis')
g_surface.set_ylabel('y-axis')

g_exp_surface = lgcp.add_subplot(222, projection='3d')
g_exp_surface.plot_surface(xv_trans_data, yv_trans_data, zv_exp, cmap='RdBu')
g_exp_surface.set_title('Exp of GRF')
g_exp_surface.set_xlabel('x-axis')
g_exp_surface.set_ylabel('y-axis')

data_plot = lgcp.add_subplot(223, projection='3d')
data_plot.scatter(xv_trans_row, yv_trans_row, histo, marker='.', color='black')
data_plot.set_title('Data Points')
data_plot.set_xlabel('x-axis')
data_plot.set_ylabel('y-axis')

plt.show()




