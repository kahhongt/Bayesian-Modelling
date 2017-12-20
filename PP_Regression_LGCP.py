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
corresponding to the observations. The resulting values of v will then form the predicted mean of the model

The hessian is the the second derivative of the log-likelihood function
Unlike standard GP Regression. LGCPs are not capable of producing predictions for any input vector, but only for inputs
corresponding to the observations. The resulting values of v will then form the predicted mean of the model.
LGCP can only predict where there has been a data point

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
    - This is tabulated at an arbitrary set of hyper-parameters - to be optimised later
b. Denominator - intractable integral
    - Laplace's Approximation
    - Bayesian Quadrature
    - MCMC Sampling
8. Tabulate optimal covariance from the Hessian
    - The matrix inverse of the Hessian is the covariance at that point - there is no explicit function
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
        else:
            if len(k_array) == len(landa_array):
                for i in range(len(prob_array)):
                    prob_array[i] = poisson_cont(k_array[i], landa_array[i])
            else:
                print('Length Mismatch')
    p_likelihood = np.prod(prob_array)  # Taking combined product of distributions
    # Note output is a scalar (singular value)
    return p_likelihood  # Returns the non logarithmic version.


def log_special(array):
    """Taking an element-wise natural log of the array, retain array dimensions"""
    """with the condition that log(0) = 0, so there are no -inf elements"""
    log_array = np.zeros(array.size)
    for i in range(array.size):
        if array[i] == 0:
            log_array[i] = 0
        else:
            log_array[i] = np.log(array[i])
    return log_array


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


# Each region is assumed to not have a constant log-intensity, uni-variate gaussian distribution assumed
def posterior_num(k_array, v_array, gaussian_mean, cov_matrix):
    """numerator consists of product of poisson distribution and distribution of v given arbitrary hyper-parameters"""
    if len(k_array) == len(v_array):
        m = len(k_array)
        exp_v_array = np.exp(v_array)  # Discretization is now removed
        """Poisson Distribution Product"""
        poisson_dist_product = poisson_product(k_array, exp_v_array)  # Uses a generalised factorial - gamma function
        """Construct latter v distribution term"""
        v_distribution_coeff = (((2 * np.pi) ** m) * np.linalg.det(cov_matrix)) ** (-0.5)
        v_distribution_power = - 0.5 * fn.matmulmul(v_array - gaussian_mean, np.linalg.inv(cov_matrix),
                                                np.transpose(v_array - gaussian_mean))
        v_distribution = v_distribution_coeff * np.exp(v_distribution_power)
        """Overall posterior numerator"""
        post_num_overall = poisson_dist_product * v_distribution

    else:
        print('k and v arrays do not have matching dimensions')

    return post_num_overall


# Generate posterior numerator function form for optimization
def posterior_numerator(param, *args):
    """We want to find the optimal v_array, so this is the parameter to be optimised"""
    v_array = param
    """args points to a tuple with 3 components, so have to create a tuple to contain them first"""
    k_array = args[0]
    gaussian_mean = args[1]
    covariance_matrix = args[2]
    posterior_numerator_value = posterior_num(k_array, v_array, gaussian_mean, covariance_matrix)
    return posterior_numerator_value


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
bins_number = 10
histo, x_edges, y_edges = np.histogram2d(x_transform, y_transform, bins=bins_number)
xv_trans_data, yv_trans_data = np.meshgrid(x_edges, y_edges)
xv_trans_data = xv_trans_data[:-1, :-1]  # Removing the last bin edge and zero points to make dimensions consistent
yv_trans_data = yv_trans_data[:-1, :-1]  # Contains a square matrix
xv_trans_row = fn.row_create(xv_trans_data)  # Creates a row from the square matrix
yv_trans_row = fn.row_create(yv_trans_data)
histo = fn.row_create(histo)
# xv_transform_row = xv_transform_row[histo != 0]  # Remove data point at histogram equal 0
# yv_transform_row = yv_transform_row[histo != 0]
# histo = histo[histo != 0]  # This is after putting them into rows

"""
Data point coordinates are now at bottom-left hand corner, coordinates of data points have
to be centralised to the centre of each quadrat
"""
# Centralizing coordinates for each quadrat
xv_trans_row = xv_trans_row + 0.5 * ((x_edges[-1] - x_edges[0]) / bins_number)
yv_trans_row = yv_trans_row + 0.5 * ((y_edges[-1] - y_edges[0]) / bins_number)

# Stack into 2 rows of many columns
xy_data_coord = np.vstack((xv_trans_row, yv_trans_row))  # location of all the data points

"""Generate auto-covariance matrix with noise - using arbitrary hyper-parameters first"""
sigma_arb = 7
length_arb = 5
noise_arb = 2
v = 3/2
# Remember arbitrary hyper-parameters were chosen

c_dd = matern_2d(v, sigma_arb, length_arb, xy_data_coord, xy_data_coord)
c_dd_noise = c_dd + (noise_arb ** 2) * np.eye(c_dd.shape[0])
# This shall be the auto-covariance matrix for the optimization

# *** The denominator of the posterior is only evaluated at the optimal vhap which is an array
# and are thus independent of the values in array v. We need to find the optimal value of v

"""Prior mean is arbitrary for now, but can be changed into a parameter afterwards"""
"""Prior mean is now taken to be the average of the sum of natural log(histo) across the entire grid"""
gp_mean_v = np.average(log_special(histo))  # Need to take log(0) = 0
# Simplified mean tabulation,otherwise the mean will have to be a parameter to be optimised


"""Find vhap, which generates the maximum of the numerator of the posterior"""
# Initialise array containing v, which also defines dimensions and value of initial parameters before optimization
v = np.zeros(xy_data_coord.shape[1])  # Taking the number of binned data points
# The above refers to the initial parameters for optimisation

arguments = (histo, gp_mean_v, c_dd_noise)

"""Define arguments and initial parameters - args, initial_param, boundary, function to be optimised"""


# Need to create a likelihood function to be minimised, which generates a set of v that is the maximum

# Note that v is an array of parameters to be optimised,
# later together with the kernel hyper-parameters


fig_lgcp = plt.figure()

data_plot = fig_lgcp.add_subplot(111, projection='3d')
data_plot.scatter(xv_trans_row, yv_trans_row, histo, marker='.', color='darkblue')

plt.show()

"""
# Select Optimization method for hyper-parameters and other conditions
opt_method = 'nelder-mead'
matern_v = 3/2  # Define matern_v
xyz_data = (xy_data_coord, histo, matern_v)
boundary_self = [(0, 30), (0, 3), (0, 3), (0, 10)]  # Can even use this into Nelder-Mead

# Optimization using self-made function
param_optimal = fn.optimise_param(opt_func=log_model_evidence, opt_arg=xyz_data, opt_method=opt_method,
                                  boundary=boundary_self)
"""

"""
# Generate Gaussian Random field - but just one example
r = np.random.randn(C_auto.shape[0], 1)  # Creates a column of random values with mean 0 and variance 1
S, V, D = np.linalg.svd(C_auto, full_matrices=True)  # Singular Value Decomposition
diagonal_V = np.diag(V)  # Constructing a diagonal matrix from the decomposed matrix
z = fn.matmulmul(S, np.sqrt(diagonal_V), r)  # This is a column containing the gaussian random surface values
zv = np.reshape(z, (x_edges[:-1].size, y_edges[:-1].size))  # Reshape to 2-D matrix form - bin in the same way as data
# Generated a Gaussian Random Field - Note that this gaussian random surface keeps changing

"""



