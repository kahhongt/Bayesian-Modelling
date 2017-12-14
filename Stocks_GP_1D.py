import matplotlib
import numpy as np
import datetime as dt
import time
import pandas as pd
import pandas_datareader as pdr
import math
import functions as fn
import matplotlib.pyplot as plt
import scipy.optimize as scopt


def mean_function(c):  # Prior mean function taken as 0 for the entire sampling range
    mean_c = np.array(np.zeros(c.size) * c)  # Element-wise multiplication
    return mean_c


def mean_func_scalar(mean, c):  # Assume that the prior mean is a constant to be optimised - non-zero mean
    if np.array([c.shape]).size == 1:
        mean_c = np.ones(1) * mean
    else:
        mean_c = np.ones(c.shape[1]) * mean
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


def mu_post(x_next, c_auto, c_cross, mismatch, p_mean):  # Posterior Mean
    if c_cross.shape[1] != c_auto.shape[1]:  # Check that the dimensions are consistent
        print('First Dimension Mismatch!')
    if c_auto.shape[0] != (np.transpose(mismatch)).shape[0]:
        print('Second Dimension Mismatch!')
    else:
        mean_post = mean_func_scalar(p_mean, x_next) + fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(mismatch))
        return mean_post


def cov_post(c_next_auto, c_cross, c_auto):  # Posterior Covariance
    c_post = c_next_auto - fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(c_cross))
    return c_post


def log_model_evidence(param, *args):  # Param includes both sigma and l, arg is passed as a pointer
    sigma = param[0]  # param is a tuple containing 2 things, which has already been defined in the function def
    length = param[1]
    noise = param[2]  # Over here we have defined each parameter in the tuple, include noise
    mean = param[3]  # The constant prior mean is now a hyper-parameter to be optimised
    x_data = args[0]  # This argument is a constant passed into the function
    y_data = args[1]
    matern_nu = args[2]
    prior_mu = mean_func_scalar(mean, x_data)
    # Prior mean function only takes in optimal mean and location of data points as inputs
    c_auto = matern(matern_nu, sigma, length, x_data, x_data)
    # c_auto = squared_exp(sigma, length, x_data, x_data)
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    c_auto_noise = c_auto + c_noise  # Overall including noise, plus include any other combination
    model_fit = - 0.5 * fn.matmulmul(y_data - prior_mu, np.linalg.inv(c_auto_noise), np.transpose(y_data - prior_mu))
    model_complexity = - 0.5 * math.log(np.linalg.det(c_auto_noise))
    model_constant = - 0.5 * len(y_data) * math.log(2*np.pi)
    log_model_evid = model_fit + model_complexity + model_constant
    return -log_model_evid  # We want to maximize the log-likelihood, meaning the min of negative log-likelihood


time_start = time.clock()  # Time start of computation

"""Importing Point Process Data Set"""
start = dt.datetime(2017, 11, 1)
end = dt.datetime(2017, 10, 1)  # Manually set end of range
present = dt.datetime.now()

apple = pdr.DataReader("AAPL", 'yahoo', start, present)  # Take note of the capitalization in DataReader
# google = pdr.DataReader("GOOGL", 'yahoo', start, end)
dt_x = (apple.index - start).days  # Have to covert to days first
x = np.array(dt_x)  # This creates an unmasked numpy array
y = apple['Adj Close'].values  # numpy.ndarray type

"""  # Resampling to take only values at certain intervals
y_length_interval = round(y.size / 1)
for i in range(y_length_interval):
    y = np.delete(y, 1 * [i])
    x = np.delete(x, 1 * [i])
"""

v = 3/2
xyv_data = (x, y, v)  # Matern function has been entered into the minimise function
initial_param = np.array([10, 10, 10, 5])  # sigma, length scale, noise
# bounds = ((0, 10), (0, 10), (0, 10))  # Hyper-parameters should be positive, Nelder-Mead does not use bounds
solution = scopt.minimize(fun=log_model_evidence, args=xyv_data, x0=initial_param, method='Nelder-Mead')
# Currently using kernel matern v = 3/2
# Nelder-mead cannot handle constraints or bounds - no bounds needed then

"""Setting Hyper-parameters""" # May sometimes be negative due to missing the target
sigma_optimal = solution.x[0]
length_optimal = solution.x[1]
noise_optimal = solution.x[2]
mean_optimal = solution.x[3]
# Here the optimal hyper-parameters are now obtained
print(solution.x)
print(solution.fun)
print(x.shape)
print(y.shape)

"""Defining entire range of potential sampling points"""
cut_off = (np.max(x) - np.min(x)) / 20
sampling_points = np.linspace(np.min(x), np.max(x) + cut_off, 200)  # Projecting 10% ahead of data set
mean_posterior = np.zeros(sampling_points.size)  # Initialise posterior mean
cov_posterior = np.zeros(sampling_points.size)  # Initialise posterior covariance
prior_mean = mean_func_scalar(mean_optimal, x)
C_dd = matern(v, sigma_optimal, length_optimal, x, x)  # v already defined above
# C_dd = squared_exp(sigma_optimal, length_optimal, x, x)
C_dd_noise = C_dd + np.eye(C_dd.shape[0]) * (noise_optimal ** 2)  # Adding optimal noise onto covariance matrix

"""Evaluating predictions for data set using optimised hyper-parameters(The next upcoming value)"""
for i in range(sampling_points.size):
    x_star = np.array([sampling_points[i]])  # make sure that I am entering an array
    C_star_d = matern(v, sigma_optimal, length_optimal, x_star, x)
    C_star_star = matern(v, sigma_optimal, length_optimal, x_star, x_star)
    # C_star_d = squared_exp(sigma_optimal, length_optimal, x_star, x)
    # C_star_star = squared_exp(sigma_optimal, length_optimal, x_star, x_star)
    prior_mismatch = y - prior_mean  # Mismatch between actual data and prior mean
    mean_posterior[i] = mu_post(x_star, C_dd_noise, C_star_d, prior_mismatch, mean_optimal)
    cov_posterior[i] = cov_post(C_star_star, C_star_d, C_dd_noise)


upper_bound = mean_posterior + (2 * np.sqrt(cov_posterior))  # Have to take the square-root of the covariance
lower_bound = mean_posterior - (2 * np.sqrt(cov_posterior))  # Showing within 3 SD of the posterior mean

time_elapsed = time.clock() - time_start
print(time_elapsed)


stock_chart = plt.figure()

"""
stock_apple = stock_chart.add_subplot(121)
stock_apple.plot(x, y, color='darkred')
stock_apple.set_title('AAPL')
stock_apple.set_xlabel('Time')
stock_apple.set_ylabel('AAPL Stock Price')
"""

pred_apple = stock_chart.add_subplot(121)
# pred_apple.plot(x, y, color='darkred', label='Actual Price', linewidth=0.5)
pred_apple.plot(sampling_points, mean_posterior, color='darkblue', label='Posterior', linewidth=0.5)
pred_apple.fill_between(sampling_points, lower_bound, upper_bound, color='lavender')  # Fill between 2-SD
pred_apple.scatter(x, y, color='darkred', label='Actual Price', s=3)
pred_apple.set_title('AAPL GP Regression')
pred_apple.set_xlabel('Time from %s to %s' % (start, present))
pred_apple.set_ylabel('AAPL Stock Posterior Distribution')

cov_apple = stock_chart.add_subplot(122)
cov_apple.plot(sampling_points, cov_posterior, color='darkblue', label='Covariance', linewidth=0.5)
cov_apple.set_title('Covariance')
cov_apple.set_xlabel('Time from %s to %s' % (start, present))
cov_apple.set_xlabel('Covariance')
plt.legend()
plt.show()
