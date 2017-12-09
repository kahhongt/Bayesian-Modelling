import matplotlib
import numpy as np
import datetime as dt
import pandas as pd
import pandas_datareader as pdr
import math
import functions as fn
import matplotlib.pyplot as plt
import scipy.optimize as scopt


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


"""Importing Point Process Data Set"""
start = dt.datetime(2016, 6, 1)
end = dt.datetime(2017, 1, 1)  # Manually set end of range
present = dt.datetime.now().date()

apple = pdr.DataReader("AAPL", 'yahoo', start, end)  # Take not of the capitalization in DataReader
# google = pdr.DataReader("GOOGL", 'yahoo', start, end)
dt_x = (apple.index - start).days  # Have to covert to days first
x = np.array(dt_x)  # This creates an unmasked numpy array
y = apple['Adj Close'].values  # numpy.ndarray type
v = 3/2

xyv_data = (x, y, v)
initial_param = np.array([10, 10, 10])  # sigma, length scale, noise
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


stock_chart = plt.figure()
stock_apple = stock_chart.add_subplot(111)
stock_apple.plot(apple['Adj Close'], color='darkblue')
stock_apple.set_title('AAPL')
stock_apple.set_xlabel('Time')
stock_apple.set_ylabel('AAPL Stock Price')
stock_apple.set_xlim(start, end)

plt.show()
