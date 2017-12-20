import numpy as np
import scipy.optimize as scopt
import matplotlib
matplotlib.use('TkAgg')


def laplace_approx(approx_func, approx_args, initial_param, approx_method):
    """Finding an approximation to the integral of the function using Laplace's Approximation"""
    # Takes in a function and imposes a gaussian function over it
    # Measure uncertainty from actual value. As M increases, the approximation of the function by
    # a gaussian function gets better. Note that an unscaled gaussian function is used.
    """Tabulate the global maximum of the function - within certain boundaries - using latin hypercube"""
    solution = scopt.minimize(fun=approx_func, arg=approx_args, x0=initial_param, method=approx_method)
    optimal_param_vect = solution.x
    optimal_func_val = solution.fun

    """Generate matrix of second derivatives - The Hessian Matrix of the function"""
    return optimal_param_vect, optimal_func_val


A = np.matrix([[1, 10, 3, 2], [8, 5, 4, 4], [2, 4, 8, 5], [3, 3, 3, 3]])
B = (1, 2)
C = (3, 4)
print(B)
print(C + B)








