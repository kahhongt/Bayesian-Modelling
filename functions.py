import numpy as np
import time
from numpy.lib import scimath
import scipy.optimize as scopt
import matplotlib
matplotlib.use('TkAgg')


"""CREATE YOUR OWN FUNCTIONS AND DEFINE THEM HERE"""


"""Simple Introductory Definitions"""


def happy_birthday():  # The function name must have a good spelling as well
    print("Happy Birthday to You")
    print("Happy Birthday to You")
    print("Happy Birthday to Me")
    print("Happy Birthday to You")


def summation(a, b):  # x and y are input arguments
    c = a + b
    return c


# print(summation(2, 1))


def array_summation(a):  # Computing the sum of a list
    q = 0  # Initialise z
    #  for i in x --- meaning for each element i in x
    for i in range(a.shape[0] + 1):  # Note range doesn't include the upper limit
        q = q + i
    return q


"""
x = np.arange(1, 6, 1)  # This is an array, so shape can be used
print(x.shape[0])
print(x)
y = np.array([1, 2, 3, 4, 5])
z = [1, 2, 3, 4, 5]  # This is a list, so shape cannot be used
print(y)  # Printing an array
print(z)  # Printing a list
print(array_summation(y))  # I can use this for x and y but not z as z is a list, not an array
"""

"""Create a function that solves a quadratic function containing coefficients a, b and c"""


def solve_quadratic(a, b, c):
    """Use the General Equation"""
    real_term = (-b)/(2*a)
    imaginary_term = (np.lib.scimath.sqrt(b**2 - 4*a*c))/(2*a)
    root1 = real_term + imaginary_term
    root2 = real_term - imaginary_term
    return root1, root2


# print(solve_quadratic(1, -1, -2))  # Yay I have created my own quadratic solver
# print(solve_quadratic(1, -1, -2)[1])  # Takes the 2nd root - remember that counter starts from 0
# print(np.lib.scimath.sqrt([-1]))
# print(array_summation(y))


"""Classes and Constructors"""
# DATA --> ATTRIBUTES
# FUNCTIONS --> METHODS


class Employee:

    def __init__(self, first, last, pay):  # This is a constructor which creates an instance of the class
        self.first = first
        self.last = last
        self.pay = pay
        self.email = first + '.' + last + '@company.com'

    def fullname(self):  # A method that takes in an instance of the class
        return '{} {}'.format(self.first, self.last)


emp_1 = Employee('Ivan', 'Payne', 100000)  # I have created an instance of an Employee class
emp_2 = Employee('Markus', 'Baxter', 50000)  # Used a method to create the email address
# print(emp_1.email)  # Methods always require parenthesis
# print(emp_2.email)
# print(emp_1.fullname())

"""Creating Switches"""


def linear(a, b):  # Takes in two rows
    a1 = np.matrix([a])  # Need to create the matrices first
    b1 = np.matrix([b])
    z = np.matmul(a1.transpose(), b1)  # Proper matrix multiplication
    return z


def f(x):
    return {
        'a': 1,
        'b': 2,
    }[x]


y = f('a')  # This assigns the value of y to be whatever defined in the function definition
# print(y)

# __init__ is used as a constructor in a class


def columnize(matrix):  # change a matrix into a column
    column = np.reshape(matrix, (matrix.size, 1))
    return column


def row_create(matrix):
    row = np.ravel(np.reshape(matrix, (1, matrix.size)))
    return row


# Triple matrix Multiplication
def matmulmul(a1, b1, c1):
    matrix_product = np.matmul(np.matmul(a1, b1), c1)
    return matrix_product


# Calculating matrix inverse using Cholesky Decomposition

# Calculating matrix inverse using LU Decomposition

# Return integer from datetime - using start and end dates
def dt_integer(dt_array, start, end):
    dt_int = dt_array.years + (dt_array.months / 12) + (dt_array.days / 365)
    return dt_int


# Generate inverse of matrix using cholesky decomposition - compare time taken with linagl.inv
def inverse_cholesky(matrix_a):
    l = np.linalg.cholesky(matrix_a)
    u = np.linalg.inv(l)
    inverse = np.matmul(u.transpose(), u)
    return inverse


# Generate artificial random stratified sample vectors using the Latin Hypercube
def initial_param_latin(bounds, guesses):  # guesses is an arbitrary input value
    final_vectors = np.zeros((len(bounds), guesses))
    while np.unique(final_vectors).size != final_vectors.size:  # While the all the elements are not unique, do the loop
        for i in range(final_vectors.shape[0]):
            for j in range(final_vectors.shape[1]):
                final_vectors[i, j] = np.random.randint(bounds[i][0] * 10, bounds[i][1] * 10) / 10
                # Generate random numbers with one decimal place
    return final_vectors


# Global Optimisation of Parameters attempt - generates the optimal parameters in the form of an array
def optimise_param(opt_func, opt_arg, opt_method, boundary):

    # note that opt_arg is a tuple containing xy_data_coord, histo and matern_v
    if opt_method == 'nelder-mead':  # Uses only an arbitrary starting point
        initial_param = np.array([10, 3, 3, 10])  # sigma, length, noise and prior mean starting point of iteration
        # No bounds needed for Nelder-Mead
        # Have to check that all values are positive
        solution = scopt.minimize(fun=opt_func, args=opt_arg, x0=initial_param, method='Nelder-Mead')
        optimal_parameters = solution.x

    elif opt_method == 'latin-hypercube-de':  # everything is already taken as an input
        solution = scopt.differential_evolution(func=opt_func, bounds=boundary, args=opt_arg,
                                                init='latinhypercube')
        optimal_parameters = solution.x

    elif opt_method == 'latin-hypercube-manual':  # manual global optimization
        guesses = 10
        ind_parameters = np.zeros((len(boundary), guesses))
        ind_func = np.zeros(guesses)
        initial_param_stacked = initial_param_latin(boundary, guesses)  # using self-made function
        for i in range(len(boundary)):
            initial_param = initial_param_stacked[:, i]
            solution = scopt.minimize(fun=opt_func, args=opt_arg, x0=initial_param, method='Nelder-Mead')
            ind_parameters[:, i] = solution.x
            ind_func[i] = solution.fun
        opt_index = np.argmin(ind_func)
        optimal_parameters = ind_parameters[:, opt_index]

    return optimal_parameters











