import matplotlib
import numpy as np
import math
import functions as fn
import matplotlib.pyplot as plt
import scipy


"""Defining the covariance and mean functions that describes the gaussian process"""


def mean_function(c):
    mean_c = np.array(np.zeros(c.size) * c)
    return mean_c


def covmatrix(kernel_inner_type, x1, x2):  # This function creates a covariance matrix with an in-built kernel function

    def kernel(f, a, b):  # Defining the covariance function within another function covmatrix
        alpha = 0.5
        if f == 'Linear':
            k = a * b
        elif f == 'Brownian':
            k = min(a, b)
        elif f == 'Squared Exponential':
            k = np.exp(-alpha * ((a - b) ** 2))
        elif f == 'Periodic':
            k = np.exp(-(np.sin(2 * np.pi * (a - b))) ** 2)  # Arbitrary setting at 2pi
        else:
            k = 1  # Arbitrarily set
        return k

    c = np.zeros((x1.size, x2.size))
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            c[i, j] = kernel(kernel_inner_type, x1[i], x2[j])  # Populating covariance matrix c
    return c  # make sure the indent is in the right place


def mu_post(x_next, c_auto, c_cross, mismatch):  # Generate posterior mean
    if c_cross.shape[1] != (np.linalg.inv(c_auto)).shape[0]:
        print('First Dimension Mismatch!')
    if (np.linalg.inv(c_auto)).shape[1] != (np.transpose(mismatch)).shape[0]:
        print('Second Dimension Mismatch!')
    else:
        mean_post = mean_function(x_next) + fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(mismatch))
        return mean_post


def cov_post(c_next_auto, c_cross, c_auto):
    c_post = c_next_auto - fn.matmulmul(c_cross, np.linalg.inv(c_auto), np.transpose(c_cross))
    return c_post


"""Creating the Data set"""
x = np.array([1, 2, 5, 10, 20, 25])
y = np.array([1, 2, 3, 5, 3, 10])  # Note that this is the data set
y_length = len(y)  # Measures number of data points

"""Define gaussian mean and kernel"""
gaussian_data_mean = mean_function(x)
kernel_stated_type = "Squared Exponential"

C_dd = covmatrix(kernel_stated_type, x, x)  # This generates the auto-covariance of x

"""Evaluating the log model evidence - without doing any predictions"""  # Only C_dd is needed here
model_fit = - 0.5 * fn.matmulmul(y - gaussian_data_mean, np.linalg.inv(C_dd), np.transpose(y - gaussian_data_mean))
model_complexity = - 0.5 * math.log(np.linalg.det(C_dd))
model_constant = - 0.5 * y_length * math.log(2*np.pi)
log_model_evidence = model_fit + model_complexity + model_constant
model_evidence = np.exp(log_model_evidence)  # Note that the model evidence isn't even used in this script
# print(model_evidence)
# print(model_fit, model_complexity, model_constant)

data = np.arange(0, 50, 0.05)  # This give us tabulation at the next predicted value
mean_posterior = np.zeros(data.size)
cov_posterior = np.zeros(data.size)

"""Evaluating predictions for data set (The next upcoming value)"""  # These are the main points to be plotted
for i in range(data.size):
    x_star = np.array([data[i]])  # making a prediction for each data point and making sure each data point is an array
    C_star_d = covmatrix(kernel_stated_type, x_star, x)  # c_cross
    C_star_star = covmatrix(kernel_stated_type, x_star, x_star)  # c_next_auto
    prior_mismatch = y - gaussian_data_mean  # mismatch between actual data and prior mean
    mean_posterior[i] = mu_post(x_star, C_dd, C_star_d, prior_mismatch)
    cov_posterior[i] = cov_post(C_star_star, C_star_d, C_dd)

upper_bound = mean_posterior + 2 * cov_posterior
lower_bound = mean_posterior - 2 * cov_posterior

fig = plt.figure()  # Creating an instance of a figure
Prediction = fig.add_subplot(111)
Prediction.scatter(x, y, c='b', marker='.')
Prediction.plot(data, mean_posterior, 'black')
Prediction.fill_between(data, lower_bound, upper_bound)

"""Setting Figure Design"""
Prediction.set_title('Gaussian Prior with Covariance ' + kernel_stated_type)
Prediction.set_xlabel('x-axis')
Prediction.set_ylabel('z-axis')
Prediction.grid(True)
plt.show()






"""

if C_star_d.shape[1] != (np.linalg.inv(C_dd)).shape[0]:
    print('First Dimension Mismatch!')
if (np.linalg.inv(C_dd)).shape[1] != (np.transpose(prior_mismatch)).shape[0]:
    print('Second Dimension Mismatch!')
else:
    mean_posterior = mean_function(x_star) + fn.matmulmul(C_star_d, np.linalg.inv(C_dd), np.transpose(prior_mismatch))
    cov_posterior = C_star_star - fn.matmulmul(C_star_d, np.linalg.inv(C_dd), np.transpose(C_star_d))

"""










