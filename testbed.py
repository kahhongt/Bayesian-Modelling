import numpy as np
import scipy.optimize as scopt
import matplotlib
import scipy.special as scispec
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


def poisson_cont(k, landa):  # to allow for non-integer k values
    numerator_p = np.power(landa, k) * np.exp(-1 * landa)
    denominator_p = scispec.gamma(k + 1)  # Generalised factorial function for non-integer k values
    # if argument into gamma function is 0, the output is a zero as well, but 0! = 1
    p = numerator_p / denominator_p
    return p


def poisson_product(k_array, landa_array):
    """Takes in 2 arrays of equal size, and takes product of poisson distributions"""
    quadrats = len(k_array)  # define the number of quadrats in total
    prob_array = np.zeros(quadrats)

    if landa_array.size == 1:
        for i in range(len(k_array)):
            prob_array[i] = poisson_cont(k_array[i], landa_array)
    else:
        if len(k_array) == len(landa_array):
            for i in range(len(prob_array)):
                prob_array[i] = poisson_cont(k_array[i], landa_array[i])
        else:
            print('Length Mismatch')
    p_likelihood = np.prod(prob_array)  # Taking combined product of distributions
    # Note output is a scalar (singular value)
    return p_likelihood  # Returns the non logarithmic version.


k_a = np.array([1, 2, 3, 4, 5, 7, 3, 2, 0, 9])
landa_a = np.array([2, 2, 2, 2, 2, 2, 2, 2, 9, 10])

print(poisson_cont(0, 2))








