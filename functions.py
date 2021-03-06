import numpy as np
from numpy.lib import scimath
import scipy.optimize as scopt
import matplotlib
matplotlib.use('TkAgg')


"""CREATE YOUR OWN FUNCTIONS AND DEFINE THEM HERE"""


"""Simple Introductory Definitions"""


def summation(a, b):  # x and y are input arguments
    c = a + b
    return c


def array_summation(a):  # Computing the sum of a list
    q = 0  # Initialise z
    #  for i in x --- meaning for each element i in x
    for i in range(a.shape[0] + 1):  # Note range doesn't include the upper limit
        q = q + i
    return q


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


def row_create(matrix):  # Generate a row from all the elements of a matrix
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


# Generate artificial random stratified sample vectors using the Latin Hypercube principle
# As the number of guesses are arbitrarily chosen, it is less effective than differential evolution, but much faster
def initial_param_latin(bounds, guesses):  # guesses is an arbitrary input value
    final_vectors = np.zeros((len(bounds), guesses))
    while np.unique(final_vectors).size != final_vectors.size:  # While the all the elements are not unique, do the loop
        for i in range(final_vectors.shape[0]):
            for j in range(final_vectors.shape[1]):
                final_vectors[i, j] = np.random.randint(bounds[i][0] * 10, bounds[i][1] * 10) / 10
                # Generate random numbers with one decimal place
    return final_vectors


# Global Optimisation of Parameters attempt - generates the optimal parameters in the form of an array
def optimise_param(opt_func, opt_arg, opt_method, boundary, initial_param):

    # note that opt_arg is a tuple containing xy_data_coord, histo and matern_v
    if opt_method == 'nelder-mead':  # Uses only an arbitrary starting point
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


def mean_func_linear(grad, intercept, c):  # Should be the correct linear regression function
    # Create array for gradient component
    if np.array([c.shape]).size == 1:
        grad_c = np.arange(0, c.size, 1)
        linear_c = (np.ones(c.size) * intercept) + (grad * grad_c)
    else:
        grad_c = np.arange(0, c.shape[1], 1)
        linear_c = (np.ones(c.shape[1]) * intercept) + (grad * grad_c)
    return linear_c


"""Numerical Integration Methods"""
# MCMC Sampling, Laplace's Approximation, Bayesian Quadrature


def hessian(matrix):  # Generates the hessian
    """Generate Hessian Matrix with finite differences - with multiple dimensions"""
    """Takes in any array/matrix and generates Hessian with 
    the dimensions (np.ndim(matrix), np.ndim(matrix), matrix.shape[0]. matrix.shape[1])"""

    matrix_grad = np.gradient(matrix)
    # Initialise Hessian - note the additional dimensions due to different direction of iteration
    hessian_matrix = np.zeros((np.ndim(matrix), np.ndim(matrix)) + matrix.shape)  # initialise correct dimensions
    for i, gradient_i in enumerate(matrix_grad):
        intermediate_grad = np.gradient(gradient_i)
        for j, gradient_ij in enumerate(intermediate_grad):
            hessian_matrix[i, j, :, :] = gradient_ij
    """Note the output will contain second derivatives in each dimensions (ii, ij, ji, jj), resulting in
    more dimensions in the hessian matrix"""
    return hessian_matrix


def jacobian(matrix):
    """Generate first derivative of a matrix - the jacobian"""
    matrix_grad = np.gradient(matrix)
    jacobian_matrix = np.zeros((np.ndim(matrix),) + matrix.shape)
    for i, gradient_i in enumerate(matrix_grad):
        jacobian_matrix[i, :, :] = gradient_i
    return jacobian_matrix


# Goal here is to estimate the denominator of the posterior
def laplace_approx(approx_func, approx_args, initial_param, approx_method):  # Note this is a general laplace approx
    """Finding an approximation to the integral of the function using Laplace's Approximation"""
    # Takes in a function and imposes a gaussian function over it
    # Measure uncertainty from actual value. As M increases, the approximation of the function by
    # a gaussian function gets better. Note that an unscaled gaussian function is used.
    """Tabulate the global maximum of the function - within certain boundaries - using latin hypercube"""
    solution = scopt.minimize(fun=approx_func, arg=approx_args, x0=initial_param, method=approx_method)
    optimal_param_vect = solution.x  # location of maximum
    optimal_func_val = solution.fun  # value at maximum of function

    """Reshape function values into a mesh grid - function takes in coordinates to generate an array of values"""



    """Tabulate the Hessian of the natural log of the function evaluated at optimal point"""


    """Generate matrix of second derivatives - The Hessian Matrix of the function"""
    return optimal_param_vect, optimal_func_val







