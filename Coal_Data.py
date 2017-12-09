import matplotlib
import numpy as np
import pandas as pd
import math
import time
import functions as fn
import matplotlib.pyplot as plt
import scipy.optimize as scopt


"""Methodology"""
# 1. Define covariance matrix and prior mean function, including the noise function
# 2. Import Data from csv into a numpy array
# 3. Convert data points into appropriately sized bins
# 4. Carry out gaussian regression by setting a covariance function
# 5. Use multivariate optimization algorithm to obtain optimal value of hyper-parameters
# 6. Carry out gaussian regression by calculating posterior mean and posterior covariance
# 7. Conduct gaussian regression for 2-D

# Note that over here I have to decouple the creation of the function and the covariance matrix

# The function must contain all the required parameters, including noise, decouple the covariance functions


def mean_function(c):  # Prior mean function taken as 0 for the entire sampling range
    mean_c = np.array(np.zeros(c.size) * c)  # Element-wise multiplication
    return mean_c


def squared_exp(sigma_exp, length_exp, x1, x2):  # Generates covariance matrix with squared exponential kernel
    c = np.zeros((x1.size, x2.size))  # ensure that the function takes in 2 arrays and not integers
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            euclidean = np.sqrt((x1[i] - x2[j]) ** 2)  # Note the square-root in the Euclidean
            exp_term = np.exp(-1 * (euclidean ** 2) * (length_exp ** -2))
            c[i, j] = (sigma_exp ** 2) * exp_term
    return c


def matern(v_value, sigma_matern, length_matern, x1, x2):  # there are only two variables in the matern function
    c = np.zeros((x1.size, x2.size))
    if v_value == 1/2:
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                euclidean = np.sqrt((x1[i] - x2[j]) ** 2)
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


def mu_post(x_next, c_auto, c_cross, mismatch):  # Posterior Mean
    if c_cross.shape[1] != (np.linalg.inv(c_auto)).shape[0]:  # Check that the dimensions are consistent
        print('First Dimension Mismatch!')
    if (np.linalg.inv(c_auto)).shape[1] != (np.transpose(mismatch)).shape[0]:
        print('Second Dimension Mismatch!')
    else:
        mean_post = mean_function(x_next) + fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(mismatch))
        return mean_post


def cov_post(c_next_auto, c_cross, c_auto):  # Posterior Covariance
    c_post = c_next_auto - fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(c_cross))
    return c_post


def log_model_evidence(param, *args):  # Param includes both sigma and l, arg is passed as a pointer
    sigma = param[0]  # param is a tuple containing 2 things, which has already been defined in the function def
    length = param[1]
    noise = param[2]  # Over here we have defined each parameter in the tuple, include noise
    x_data = args[0]  # This argument is a constant passed into the function
    y_data = args[1]
    matern_nu = args[2]
    prior_mu = mean_function(x_data)
    c_auto = matern(matern_nu, sigma, length, x_data, x_data)
    # c_auto = squared_exp(sigma, length, x_data, x_data)
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    c_auto_noise = c_auto + c_noise  # Overall including noise, plus include any other combination
    model_fit = - 0.5 * fn.matmulmul(y_data - prior_mu, np.linalg.inv(c_auto_noise), np.transpose(y_data - prior_mu))
    model_complexity = - 0.5 * math.log(np.linalg.det(c_auto_noise))
    model_constant = - 0.5 * len(y_data) * math.log(2*np.pi)
    log_model_evid = model_fit + model_complexity + model_constant
    return -log_model_evid  # We want to maximize the log-likelihood, meaning the min of negative log-likelihood


start_time = time.clock()

"""Importing Point Process Data Set"""
A = np.genfromtxt('Coal_Data_Bayesian.csv', delimiter=',')  # Extract from csv using numpy
df = pd.read_csv('Coal_Data_Bayesian.csv')  # Generates a DataFrame from csv - coal data
df_array = np.ravel(df)  # Conversion into a numpy array containing the data points

"""Auto Bin Size - Optimal Bin Width Selection"""
df_tuple_auto = np.histogram(df_array, bins='auto')  # [0] gives the histogram, [1] gives the bin edges
# df_tuple_fd = np.histogram(df_array, bins='fd')  # Freedman Diaconis Estimator-robust-considers variability and size
df_intervals = df_tuple_auto[1][:-1]  # Removing the last edge so that array size is consistent
df_histogram_auto = df_tuple_auto[0]  # Histogram values

"""Sample Data Set"""
# x = np.array([1, 2, 3, 4, 7, 8, 9, 10])
# y = np.array([2, 3, 5, 8, 15, 8, 10, 7])  # not used yet

x = df_intervals  # Data set intervals
y = df_histogram_auto  # Data set values
v = 3/2


xyv_data = (x, y, v)
initial_param = np.array([50, 50, 50])  # sigma, length scale, noise
# bounds = ((0, 10), (0, 10), (0, 10))  # Hyper-parameters should be positive, Nelder-Mead does not use bounds
solution = scopt.minimize(fun=log_model_evidence, args=xyv_data, x0=initial_param, method='Nelder-Mead')

# Nelder-mead cannot handle constraints or bounds - no bounds needed then

"""Setting Hyper-parameters""" # May sometimes be negative due to missing the target
sigma_optimal = solution.x[0]
length_optimal = solution.x[1]
noise_optimal = solution.x[2]
# Here I have obtained the optimal hyper-parameters
print(sigma_optimal, length_optimal, noise_optimal)
log_likelihood = log_model_evidence(solution.x, *xyv_data)
print(log_likelihood)


"""Defining entire range of potential sampling points"""
cut_off = (np.max(x) - np.min(x)) / 100
sampling_points = np.linspace(np.min(x) - cut_off, np.max(x) + cut_off, 100)
mean_posterior = np.zeros(sampling_points.size)  # Initialise posterior mean
cov_posterior = np.zeros(sampling_points.size)  # Initialise posterior covariance
prior_mean = mean_function(x)
C_dd = matern(v, sigma_optimal, length_optimal, x, x)
# C_dd = squared_exp(sigma_optimal, length_optimal, x, x)
C_dd_noise = C_dd + np.eye(C_dd.shape[0]) * (noise_optimal ** 2)

"""Evaluating predictions for data set using optimised hyper-parameters(The next upcoming value)"""
for i in range(sampling_points.size):
    x_star = np.array([sampling_points[i]])  # make sure that I am entering an array
    C_star_d = matern(v, sigma_optimal, length_optimal, x_star, x)
    C_star_star = matern(v, sigma_optimal, length_optimal, x_star, x_star)
    # C_star_d = squared_exp(sigma_optimal, length_optimal, x_star, x)
    # C_star_star = squared_exp(sigma_optimal, length_optimal, x_star, x_star)
    prior_mismatch = y - prior_mean  # Mismatch between actual data and prior mean
    mean_posterior[i] = mu_post(x_star, C_dd_noise, C_star_d, prior_mismatch)
    cov_posterior[i] = cov_post(C_star_star, C_star_d, C_dd_noise)


upper_bound = mean_posterior + (2 * np.sqrt(cov_posterior))  # Have to take the square-root of the covariance
lower_bound = mean_posterior - (2 * np.sqrt(cov_posterior))

a = np.array([[1, 2, 3, 4], [2, 2, 2, 2], [1, 1, 1, 1]])
b = np.array([1, 2, 3, 4])
print(a)
print(a.shape)
print(np.array([a.shape]).size)
print(b)
print(b.shape)
print(np.array([b.shape]).size)


"""Start the plot"""

fig_coal = plt.figure()  # Creating an instance of a figure
Prediction = fig_coal.add_subplot(221)
Prediction.plot(sampling_points, mean_posterior, 'darkblue')
Prediction.fill_between(sampling_points, lower_bound, upper_bound, color='skyblue')  # Fill between 2-SD
Prediction.scatter(x, y, color='black', marker='.')
Prediction.set_title('Posterior Distribution '+'Matern ' + '%s' % v)
Prediction.set_xlabel('x-axis')
Prediction.set_ylabel('y-axis')
Prediction.set_xlim(np.min(sampling_points), np.max(sampling_points))
Prediction.set_ylim(np.min(lower_bound), np.max(upper_bound))
Prediction.grid(True)

Coal_data = fig_coal.add_subplot(222)
Coal_data.scatter(x, y, color='black', marker='.')
Coal_data.set_title('Coal Data Scatter Plot')
Coal_data.set_xlabel('x-axis')
Coal_data.set_ylabel('y-axis')
Coal_data.set_xlim(np.min(sampling_points), np.max(sampling_points))
Coal_data.set_ylim(np.min(lower_bound), np.max(upper_bound))
Coal_data.grid(True)

elapsed_time = time.clock() - start_time
print(elapsed_time)

plt.show()


"""
# Coal_hist_manual = fig.add_subplot(221)
# Coal_hist_manual.scatter(df_tuple_manual[1][:-1], df_tuple_manual[0])

Coal_hist_auto = fig.add_subplot(222)
Coal_hist_auto.scatter(df_tuple_auto[1][:-1], df_tuple_auto[0])

# Coal_hist_fd = fig.add_subplot(223)
# Coal_hist_fd.scatter(df_tuple_fd[1][:-1], df_tuple_fd[0])



# Interesting to explore pandas.cut further
# df_bin_alloc = pd.cut(df_array, 10, right=True, include_lowest=True)  # Sorting data into ranges

# Manually-selected Bin Size - For Control Purposes
df_bins_manual = np.linspace(df_array.min(), df_array.max(), 20)  # Creating the bins
df_tuple_manual = np.histogram(df_array, df_bins_manual)  # Successfully counting the number in each bin
# The tuple output above contains 2 arrays, have to extract the histogram using df_histogram[0]
df_histogram_manual = df_tuple_manual[0]
df_intervals = df_bins_manual[:-1]  # Removing the last edge of bins for plotting (intervals indicate left edge of bin)



"""


