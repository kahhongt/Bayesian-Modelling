import matplotlib
import numpy as np
import pandas as pd
import math
import functions as fn
import matplotlib.pyplot as plt
import scipy.optimize as scopt


def mean_function(c):  # Prior mean function taken to be 0
    mean_c = np.zeros(c.shape)
    return mean_c


def squared_exp(sigma_exp, length_exp, x1, x2):
    c = np.zeros((x1.size, x2.size))
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            euclidean = np.sqrt((x1[i] - x2[j]) ** 2)
            exp_term = np.exp(-1 * (euclidean ** 2) * (length_exp ** -2))
            c[i, j] = (sigma_exp ** 2) * exp_term
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
    prior_mu = mean_function(x_data)
    c_auto = squared_exp(sigma, length, x_data, x_data)
    # c_auto = matern(3/2, sigma, length, x_data, x_data)  # Select kernel here to optimize
    c_noise = np.eye(c_auto.shape[0]) * (noise ** 2)  # Fro-necker delta function
    c_auto_noise = c_auto + c_noise  # Overall including noise, plus include any other combination
    model_fit = - 0.5 * fn.matmulmul(y_data - prior_mu, np.linalg.inv(c_auto_noise), np.transpose(y_data - prior_mu))
    model_complexity = - 0.5 * math.log(np.linalg.det(c_auto_noise))
    model_constant = - 0.5 * len(y_data) * math.log(2*np.pi)
    log_model_evid = model_fit + model_complexity + model_constant
    return -log_model_evid  # We want to maximize the log-likelihood, meaning the min of negative log-likelihood


x = np.array([1, 2, 3, 4, 7, 8, 9, 10])
y = np.array([2, 3, 5, 8, 15, 8, 10, 7])

xy_data = (x, y)
initial_param = np.array([10, 10, 10])  # sigma, length scale, noise
bounds = ((1, 10), (1, 10), (0, 10))  # Hyper-parameters should be positive
solution = scopt.minimize(fun=log_model_evidence, args=xy_data, x0=initial_param, method='Nelder-mead')



print(solution.x)

fig_coal= plt.figure()
Graph = fig_coal.add_subplot(111)
Graph.scatter(x, y, color='black', marker='.')
plt.show()

