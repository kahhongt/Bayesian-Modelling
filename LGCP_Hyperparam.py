import pandas as pd
import math
import matplotlib
import numpy as np
import time
import functions as fn
import scipy
import scipy.optimize as scopt
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Bayesian Method of Evaluating the Posterior.
The standard Gaussian Process Regression with optimised hyperparameters by maximising the
marginal likelihood only applies when there is not latent gaussian intensity function that
describes each of the data set values

Each quadrat is now assumed to have a number of occurrences according to a latent gaussian
 field which defines the intensity function at each quadrat.
 
The latent field x behaves according to a function of a gaussian process (in this case,
the function is the exponential of the latent gaussian process. The latent gaussian process 
is assumed to have zero mean and a covariance matrix parameterised by theta, where theta refers
to the set of hyper-parameters

1. Define the Poisson Distribution for each quadrat, even for quadrats without any occurrences
2. Obtain the expression for likelihood, based on the product of the poisson distributions
for all the quadrats
3. Using the marginal likelihood, we can obtain the calculate the value of x, x0 that leads to
the greatest marginal likelihood.
4. However, to obtain the poisson distribution for each likelihood, we need to know the latent field x

"""


# Define Poisson Distribution function for each quadrat
def poisson(k, landa):  # Takes in two parameters intensity landa and observation value k
    numerator = np.power(landa, k) * np.exp(-1 * landa)  # mean and covariance are both landa
    denominator = math.factorial(k)
    p = numerator / denominator
    return p


# Create likelihood for all quadrats
def poisson_likelihood(k_array, landa_array):
    prob_array = np.zeros(len(k_array))
    if len(k_array) == len(landa_array):
        for i in range(len(prob_array)):
            prob_array[i] = poisson(k_array[i], landa_array[i])
        p_likelihood = np.prod(prob_array)
    else:
        print('Length Mismatch')
    return p_likelihood


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
xv_transform, yv_transform = np.meshgrid(x_edges, y_edges)
xv_transform = xv_transform[:-1, :-1]  # Removing the last bin edge and zero points to make dimensions consistent
yv_transform = yv_transform[:-1, :-1]  # Contains a square matrix
xv_transform_row = fn.row_create(xv_transform)  # Creates a row from the square matrix
yv_transform_row = fn.row_create(yv_transform)
histo = fn.row_create(histo)
# xv_transform_row = xv_transform_row[histo != 0]  # Remove data point at histogram equal 0
# yv_transform_row = yv_transform_row[histo != 0]
# histo = histo[histo != 0]  # This is after putting them into rows
xy_data_coord = np.vstack((xv_transform_row, yv_transform_row))  # includes all the zero points


"""Generate Gaussian Process Prior w to obtain the latent model"""





