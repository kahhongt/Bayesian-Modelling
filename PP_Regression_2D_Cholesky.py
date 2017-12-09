import pandas as pd
import math
import matplotlib
import numpy as np
import time
import functions as fn
import scipy.optimize as scopt
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""Methodology for Conducting Gaussian Regression for 2-D"""
# 1. Using Matern 3/2, calculate hyper-parameters for maximum likelihood (evidence)
# 2. Use Nelder-Mead first, then use the Hypercube to obtain the globally optimal hyper-parameters
# 3. Make sure you understand the matrix manipulation for the 2-D GP
# 4. Create an arbitrary transformation matrix which can also optimised before hyper-parameters
# 5. Using csv.read, import the 2-D point process data and plot it first


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
        mean_post = mean_func_scalar(p_mean, xy_next) + fn.matmulmul(c_cross, fn.inverse_cholesky(c_auto), np.transpose(mismatch))
        return mean_post


def cov_post(c_next_auto, c_cross, c_auto):  # Posterior Covariance
    c_post = c_next_auto - fn.matmulmul(c_cross, fn.inverse_cholesky(c_auto), np.transpose(c_cross))
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
    model_fit = - 0.5 * fn.matmulmul(histogram_data - prior_mu, fn.inverse_cholesky(c_auto_noise),
                                     np.transpose(histogram_data - prior_mu))
    model_complexity = - 0.5 * math.log(np.linalg.det(c_auto_noise))
    model_constant = - 0.5 * len(histogram_data) * math.log(2*np.pi)
    log_model_evid = model_fit + model_complexity + model_constant
    return -log_model_evid  # We want to maximize the log-likelihood, meaning the min of negative log-likelihood


time_start = time.clock()

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
xv_transform, yv_transform = np.meshgrid(x_edges, y_edges)
xv_transform = xv_transform[:-1, :-1]  # Removing the last bin edge and zero points to make dimensions consistent
yv_transform = yv_transform[:-1, :-1]  # Contains a square matrix
xv_transform_row = fn.row_create(xv_transform)  # Creates a row from the square matrix
yv_transform_row = fn.row_create(yv_transform)
histo = fn.row_create(histo)
# xv_transform_row = xv_transform_row[histo != 0]  # Remove data point at histogram equal 0
# yv_transform_row = yv_transform_row[histo != 0]
# histo = histo[histo != 0]  # This is after putting them into rows
xy_data_coord = np.vstack((xv_transform_row, yv_transform_row))


"""Calculate optimal hyper-parameters using Nelder-Mead with Initial Parameters"""
# Select Optimization method for hyper-parameters
opt_method = 'nelder-mead'
matern_v = 3/2  # Define matern_v
xyz_data = (xy_data_coord, histo, matern_v)

if opt_method == 'nelder-mead':
    initial_param = np.array([10, 3, 3, 10])  # sigma, length, noise and prior mean starting point of iteration
    # No bounds needed for Nelder-Mead
    # Have to check that all values are positive
    solution = scopt.minimize(fun=log_model_evidence, args=xyz_data, x0=initial_param, method='Nelder-Mead')

elif opt_method == 'latin-hypercube':
    boundary = [(0, 30), (0, 3), (0, 3), (0, 10)]
    solution = scopt.differential_evolution(func=log_model_evidence, bounds=boundary, args=xyz_data,
                                            init='latinhypercube')


sigma_optimal = solution.x[0]
length_optimal = solution.x[1]
noise_optimal = solution.x[2]
mean_optimal = solution.x[3]
print(solution)
print(mean_optimal)
print('mean of data set = %s' % np.average(histo))
print(histo.shape)
print(xy_data_coord.shape)


"""Defining the entire range of potential sampling points"""
intervals = 50  # no. of intervals on each axis
cut_off_x = (np.max(xv_transform_row) - np.min(xv_transform_row)) / 5
cut_off_y = (np.max(yv_transform_row) - np.min(yv_transform_row)) / 5
sampling_points_x = np.linspace(np.min(xv_transform_row) - cut_off_x, np.max(xv_transform_row) + cut_off_x, intervals)
sampling_points_y = np.linspace(np.min(yv_transform_row) - cut_off_y, np.max(yv_transform_row) + cut_off_y, intervals)
# Create iteration for coordinates using mesh-grid
sampling_points_xmesh, sampling_points_ymesh = np.meshgrid(sampling_points_x, sampling_points_y)
sampling_x_row = fn.row_create(sampling_points_xmesh)
sampling_y_row = fn.row_create(sampling_points_ymesh)
sampling_coord = np.vstack((sampling_x_row, sampling_y_row))


"""Initialise posterior mean and posterior covariance"""
mean_posterior = np.zeros(sampling_coord.shape[1])
cov_posterior = np.zeros(sampling_coord.shape[1])
# Prior mean tabulated from data set, not sampling points - now assume a non-zero mean
prior_mean = mean_func_scalar(mean_optimal, xy_data_coord)  # should be one row of zeros even though data has two rows
# The prior mean is now non-zero - the last element of the argument set

"""Create auto-covariance matrix"""
C_dd = matern_2d(matern_v, sigma_optimal, length_optimal, xy_data_coord, xy_data_coord)
C_noise = np.eye(C_dd.shape[0]) * (noise_optimal ** 2)
C_dd_noise = C_dd + C_noise
C_dd_noise_row = fn.row_create(C_dd_noise)
prior_mismatch = histo - prior_mean
xv_mesh, yv_mesh = np.meshgrid(xv_transform_row, yv_transform_row)
xv_mesh_row = fn.row_create(xv_mesh)
yv_mesh_row = fn.row_create(yv_mesh)


"""Tabulating the posterior mean and covariance for each sampling point"""
for i in range(sampling_coord.shape[1]):
    xy_star = sampling_coord[:, i]
    C_star_d = matern_2d(matern_v, sigma_optimal, length_optimal, xy_star, xy_data_coord)
    C_star_star = matern_2d(matern_v, sigma_optimal, length_optimal, xy_star, xy_star)
    mean_posterior[i] = mu_post(mean_optimal, xy_star, C_dd_noise, C_star_d, prior_mismatch)
    cov_posterior[i] = cov_post(C_star_star, C_star_d, C_dd_noise)


"""Creating 2-D inputs for plotting surfaces"""
sampling_x_2d = sampling_x_row.reshape(intervals, intervals)
sampling_y_2d = sampling_y_row.reshape(intervals, intervals)
mean_posterior_2d = mean_posterior.reshape(intervals, intervals)
cov_posterior_2d = cov_posterior.reshape(intervals, intervals)


"""Plot Settings"""
fig_pp = plt.figure(figsize=(20, 10))  # Define size of figure

data_transform = fig_pp.add_subplot(231)  # Transformed Data Scatter
data_transform.scatter(x_transform, y_transform, color='darkblue', marker='.')
data_transform.set_title('Rotated Data Set' + '(Theta = %s) ' % theta)
data_transform.set_xlabel('x-axis')
data_transform.set_ylabel('y-axis')
data_transform.grid(True)

bin_plot = fig_pp.add_subplot(232, projection='3d')
bin_plot.scatter(xv_transform_row, yv_transform_row, histo, color='darkblue', marker='.')
bin_plot.set_title('3-D Binned Plot')
bin_plot.set_xlabel('x-axis')
bin_plot.set_ylabel('y-axis')
bin_plot.grid(True)

post_mean_plot = fig_pp.add_subplot(233, projection='3d')
post_mean_plot.plot_surface(sampling_x_2d, sampling_y_2d, mean_posterior_2d, cmap='RdBu')
# post_mean_plot.plot_wireframe(sampling_x_2d, sampling_y_2d, mean_posterior_2d, color='darkblue')
post_mean_plot.set_title('Posterior Mean 3D-Plot')
post_mean_plot.set_xlabel('x-axis')
post_mean_plot.set_ylabel('y-axis')
post_mean_plot.set_zlabel('mean-axis')
post_mean_plot.grid(True)

post_cov_plot = fig_pp.add_subplot(234, projection='3d')
post_cov_plot.plot_surface(sampling_x_2d, sampling_y_2d, cov_posterior_2d, cmap='RdBu')
post_cov_plot.set_title('Posterior Covariance 3D-Plot')
post_cov_plot.set_xlabel('x-axis')
post_cov_plot.set_ylabel('y-axis')
post_cov_plot.set_zlabel('covariance-axis')
post_cov_plot.grid(True)

post_mean_color = fig_pp.add_subplot(235)
post_mean_color.pcolor(sampling_points_x, sampling_points_y, mean_posterior_2d, cmap='RdBu')
post_mean_color.set_title('Posterior Mean Color Map')
post_mean_color.set_xlabel('x-axis')
post_mean_color.set_ylabel('y-axis')
post_mean_color.grid(True)

post_cov_color = fig_pp.add_subplot(236)
post_cov_color.pcolor(sampling_points_x, sampling_points_y, cov_posterior_2d, cmap='RdBu')
post_cov_color.set_title('Posterior Covariance Color Map')
post_cov_color.set_xlabel('x-axis')
post_cov_color.set_ylabel('y-axis')
post_cov_color.grid(True)

time_elapsed = time.clock() - time_start
print(time_elapsed)

plt.show()



"""
data_transform = fig_pp_data.add_subplot(121)  # Transformed Data Scatter
data_transform.scatter(x_transform, y_transform, color='darkblue', marker='.')
data_transform.set_title('Rotated Data Set' + '(Theta = %s) ' % theta)
data_transform.set_xlabel('x-axis')
data_transform.set_ylabel('y-axis')
data_transform.grid(True)

bin_plot = fig_pp_data.add_subplot(122, projection='3d')
bin_plot.scatter(xv_transform_row, yv_transform_row, histo, color='darkblue', marker='.')
bin_plot.set_title('3-D Binned Plot')
bin_plot.set_xlabel('x-axis')
bin_plot.set_ylabel('y-axis')
bin_plot.grid(True)

fig_pp_pred = plt.figure(2)

post_mean_plot = fig_pp_pred.add_subplot(221, projection='3d')
post_mean_plot.plot_surface(sampling_x_2d, sampling_y_2d, mean_posterior_2d, cmap='RdBu')
# post_mean_plot.plot_wireframe(sampling_x_2d, sampling_y_2d, mean_posterior_2d, color='darkblue')
post_mean_plot.set_title('Posterior Mean 3D-Plot')
post_mean_plot.set_xlabel('x-axis')
post_mean_plot.set_ylabel('y-axis')
post_mean_plot.set_zlabel('mean-axis')
post_mean_plot.grid(True)

post_cov_plot = fig_pp_pred.add_subplot(222, projection='3d')
post_cov_plot.plot_surface(sampling_x_2d, sampling_y_2d, cov_posterior_2d, cmap='RdBu')
post_cov_plot.set_title('Posterior Covariance 3D-Plot')
post_cov_plot.set_xlabel('x-axis')
post_cov_plot.set_ylabel('y-axis')
post_cov_plot.set_zlabel('covariance-axis')
post_cov_plot.grid(True)

post_mean_color = fig_pp_pred.add_subplot(223)
post_mean_color.pcolor(sampling_points_x, sampling_points_y, mean_posterior_2d, cmap='RdBu')
post_mean_color.set_title('Posterior Mean Color Map')
post_mean_color.set_xlabel('x-axis')
post_mean_color.set_ylabel('y-axis')
post_mean_color.grid(True)

post_cov_color = fig_pp_pred.add_subplot(224)
post_cov_color.pcolor(sampling_points_x, sampling_points_y, cov_posterior_2d, cmap='RdBu')
post_cov_color.set_title('Posterior Covariance Color Map')
post_cov_color.set_xlabel('x-axis')
post_cov_color.set_ylabel('y-axis')
post_cov_color.grid(True)

wireframes = plt.figure(3)

post_mean_wire = wireframes.add_subplot(121, projection='3d')
post_mean_wire.plot_surface(sampling_x_2d, sampling_y_2d, mean_posterior_2d, cmap='RdBu')
post_mean_wire.set_title('Posterior Mean')
post_mean_wire.set_xlabel('x-axis')
post_mean_wire.set_ylabel('y-axis')
post_mean_wire.grid(True)

post_cov_wire = wireframes.add_subplot(122, projection='3d')
post_cov_wire.plot_surface(sampling_x_2d, sampling_y_2d, cov_posterior_2d, cmap='RdBu')
post_cov_wire.set_title('Posterior Covariance')
post_cov_wire.set_xlabel('x-axis')
post_cov_wire.set_ylabel('y-axis')
post_cov_wire.grid(True)
"""

""" Plotting the Kernel matrix in x and y coordinates
kernel = plt.figure(4)
kernel_plot = kernel.add_subplot(111, projection='3d')
# kernel_plot.scatter(xv_mesh_row, yv_mesh_row, C_dd_noise_row)
kernel_plot.plot_wireframe(xv_mesh, yv_mesh, C_dd_noise, cmap='RdBu')
kernel_plot.set_xlabel('x-axis')
kernel_plot.set_ylabel('y-axis')
kernel_plot.grid(True)
"""




