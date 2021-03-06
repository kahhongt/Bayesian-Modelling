import pandas as pd
import math
import matplotlib
import numpy as np
import functions as fn
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy.optimize as scopt

"""Methodology for Conducting Gaussian Regression for 2-D"""
# 1.
# 1. Using Matern 3/2, calculate hyper-parameters for maximum likelihood (evidence)
# 2. Use Nelder-Mead first, then use the Hypercube to obtain the globally optimal hyper-parameters
# 3. Make sure you understand the matrix manipulation for the 2-D GP
# 4. Create an arbitrary transformation matrix which can also optimised before hyper-parameters
# 5. Using csv.read, import the 2-D point process data and plot it first


def mean_func_zero(c):  # Prior mean function taken as 0 for the entire sampling range
    mean_c = np.zeros(c.shape[1])  # Element-wise multiplication
    return mean_c  # Outputs a x and y coordinates, created from the mesh grid


def squared_exp(sigma_exp, length_exp, x1, x2):  # Generates covariance matrix with squared exponential kernel
    # Define horizontal and vertical dimensions of covariance matrix c

    if np.array([x1.shape]).size == 1 and np.array([x2.shape]).size != 1:
        rows = x1.size
        columns = x2.shape[1]
    elif np.array([x1.shape]).size != 1 and np.array([x2.shape]).size == 1:
        rows = x1.shape[1]
        columns = x2.size
    elif np.array([x1.shape]).size == 1 and np.array([x2.shape]).size == 1:
        rows = x1.size
        columns = x2.size
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


def matern(v_value, sigma_matern, length_matern, x1, x2):  # there are only two variables in the matern function
    c = np.zeros((x1.size, x2.size))
    if v_value == 1/2:
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                euclidean = np.sqrt((x1[i] - x2[j]) ** 2)  # Have to change to 2-D
                exp_term = np.exp(-1 * euclidean * (length_matern ** -1))
                c[i, j] = (sigma_matern ** 2) * exp_term

    if v_value == 3/2:
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                euclidean = np.sqrt((x1[i] - x2[j]) ** 2)
                coefficient_term = (1 + np.sqrt(3) * euclidean * (length_matern ** -1))
                exp_term = np.exp(-1 * np.sqrt(3) * euclidean * (length_matern ** -1))
                c[i, j] = (sigma_matern ** 2) * coefficient_term * exp_term
    return c
# Both kernel functions take in numpy arrays of one row (create a single column first)


def mu_post(xy_next, c_auto, c_cross, mismatch):  # Posterior mean
    if c_cross.shape[1] != (np.linalg.inv(c_auto)).shape[0]:
        print('First Dimension Mismatch!')
    if (np.linalg.inv(c_auto)).shape[1] != (np.transpose(mismatch)).shape[0]:
        print('Second Dimension Mismatch!')
    else:
        mean_post = mean_func_zero(xy_next) + fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(mismatch))
        return mean_post


def cov_post(c_next_auto, c_cross, c_auto):  # Posterior Covariance
    c_post = c_next_auto - fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(c_cross))
    return c_post


def log_model_evidence(param, *args):  # Param includes both sigma and l, arg is passed as a pointer
    sigma = param[0]  # param is a tuple containing 2 things, which has already been defined in the function def
    length = param[1]
    noise = param[2]  # Over here we have defined each parameter in the tuple, include noise
    xy_coordinates = args[0]  # This argument is a constant passed into the function
    histogram_data = args[1]  # Have to enter histogram data as well
    prior_mu = mean_func_zero(xy_coordinates)  # This creates a matrix with 2 rows
    # c_auto = matern(matern_nu, sigma, length, x_data, x_data)
    c_auto = squared_exp(sigma, length, xy_coordinates, xy_coordinates)
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    c_auto_noise = c_auto + c_noise  # Overall including noise, plus include any other combination
    model_fit = - 0.5 * fn.matmulmul(histogram_data - prior_mu, np.linalg.inv(c_auto_noise),
                                     np.transpose(histogram_data - prior_mu))
    model_complexity = - 0.5 * math.log(np.linalg.det(c_auto_noise))
    model_constant = - 0.5 * len(histogram_data) * math.log(2*np.pi)
    log_model_evid = model_fit + model_complexity + model_constant
    return -log_model_evid  # We want to maximize the log-likelihood, meaning the min of negative log-likelihood


"""Extract Data from csv"""  # Arbitrary Point Process Data
A = np.genfromtxt('PP_Data_2D.csv', delimiter=',')  # Extract from csv using numpy
df = pd.read_csv('PP_Data_2D.csv')  # Generates a DataFrame from csv - coal data
x = np.ravel(df.values[0])
y = np.ravel(df.values[1])


"""Specify rotation matrix for data set"""
theta = 0  # Specify degree of rotation
mat_transform = np.matrix([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
df_matrix = np.vstack((x, y))
df_transform = mat_transform * df_matrix
x_transform = np.ravel(df_transform[0])
y_transform = np.ravel(df_transform[1])


"""Bin point process data"""
histo, x_edges, y_edges = np.histogram2d(x_transform, y_transform, bins=10)
xv_transform, yv_transform = np.meshgrid(x_edges, y_edges)
xv_transform = xv_transform[:-1, :-1]  # Removing the last bin edge and zero points to make dimensions consistent
yv_transform = yv_transform[:-1, :-1]  # Contains a square matrix
xv_transform_row = fn.row_create(xv_transform)  # Creates a row from the square matrix
yv_transform_row = fn.row_create(yv_transform)
histo = fn.row_create(histo)
xv_transform_row = xv_transform_row[histo != 0]  # Remove data point at histogram equal 0
yv_transform_row = yv_transform_row[histo != 0]
xy_data_coord = np.vstack((xv_transform_row, yv_transform_row))
histo = histo[histo != 0]  # This is after putting them into rows


"""Calculate optimal hyper-parameters"""
xyz_data = (xy_data_coord, histo)
initial_param = np.array([10, 2, 3])  # sigma, length and noise - find a good one to reduce iterations
# No bounds needed for Nelder-Mead
solution = scopt.minimize(fun=log_model_evidence, args=xyz_data, x0=initial_param, method='Nelder-Mead')
sigma_optimal = solution.x[0]
length_optimal = solution.x[1]
noise_optimal = solution.x[2]
log_evidence_optimal = log_model_evidence(solution.x, *xyz_data)  # This computes the optimal log-likelihood


"""Defining the entire range of potential sampling points"""
cut_off_x = (np.max(xv_transform_row) - np.min(xv_transform_row)) / 100
cut_off_y = (np.max(yv_transform_row) - np.min(yv_transform_row)) / 100
sampling_points_x = np.linspace(np.min(xv_transform_row), np.max(xv_transform_row), 50)
sampling_points_y = np.linspace(np.min(yv_transform_row), np.max(yv_transform_row), 50)
# Create iteration for coordinates using mesh-grid
sampling_points_xmesh, sampling_points_ymesh = np.meshgrid(sampling_points_x, sampling_points_y)
sampling_x_row = fn.row_create(sampling_points_xmesh)
sampling_y_row = fn.row_create(sampling_points_ymesh)
sampling_coord = np.vstack((sampling_x_row, sampling_y_row))


"""Initialise posterior mean and posterior covariance"""
mean_posterior = np.zeros(sampling_coord.shape[1])
cov_posterior = np.zeros(sampling_coord.shape[1])
# Prior mean tabulated from data set, not sampling points
prior_mean = mean_func_zero(xy_data_coord)  # should be one row of zeros even though data has two rows


"""Create auto-covariance matrix"""
C_dd = squared_exp(sigma_optimal, length_optimal, xy_data_coord, xy_data_coord)
C_noise = np.eye(C_dd.shape[0]) * (noise_optimal ** 2)
C_dd_noise = C_dd + C_noise
prior_mismatch = histo - prior_mean
print(sampling_coord[:, 2])
print(sampling_coord[:, 2].shape)
print(sampling_coord.shape)


"""Evaluate posterior mean and covariance"""
for i in range(sampling_coord.shape[1]):
    xy_star = sampling_coord[:, i]
    # Has to tabulate covariance matrix for each value
    C_star_d = squared_exp(sigma_optimal, length_optimal, xy_star, xy_data_coord)
    C_star_star = squared_exp(sigma_optimal, length_optimal, xy_star, xy_star)
    mean_posterior[i] = mu_post(xy_star, C_dd_noise, C_star_d, prior_mismatch)
    cov_posterior[i] = cov_post(C_star_star, C_star_d, C_dd_noise)


print(mean_posterior)
print(cov_posterior)


"""
fig_pp = plt.figure()
data_original = fig_pp.add_subplot(221)  # Original Data Scatter
data_original.scatter(x, y, color='darkblue', marker='.')
data_original.set_title('Original Data Set')
data_original.set_xlabel('x-axis')
data_original.set_ylabel('y-axis')
data_original.grid(True)

data_transform = fig_pp.add_subplot(222)  # Transformed Data Scatter
data_transform.scatter(x_transform, y_transform, color='darkblue', marker='.')
data_transform.set_title('Rotated Data Set')
data_transform.set_xlabel('x-axis')
data_transform.set_ylabel('y-axis')
data_transform.grid(True)

bin_plot = fig_pp.add_subplot(223, projection='3d')
# bin_plot.plot_surface(xv_transform, yv_transform, H, cmap=cm.coolwarm)
bin_plot.scatter(xv_transform_row[histo != 0], yv_transform_row[histo != 0], histo[histo != 0], color='darkblue', marker='.')
bin_plot.set_title('3-D Binned Plot')
plt.show()

"""