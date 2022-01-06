import numpy as np
from scipy.optimize import minimize_scalar

##############################################################################

def golden_vec_search(low_bound, up_bound, f):

    """
    Executes the golden section search algorithm for minimizing the function f
    at each lower bound and upper bound pair in parallel.

    :param low_bound: A NumPy array of lower bounds for the optimization algo.
    :param up_bound: A NumPy array of upper bounds for the optimization algo.
    :param f: The function to be minimized

    :return: A NumPy array of the optimal points found for all the bounds 
    """

    R = (5**0.5 - 1) / 2
    D = R * (up_bound - low_bound)
    x1 = low_bound + D
    x2 = up_bound - D
    f1 = f(x1)
    f2 = f(x2)

    for i in range(50):

        f1_low = (f1 < f2) * 1
        f2_low = (f1_low - 1) * (-1)

        low_bound = x2 * f1_low + low_bound * f2_low
        x2 = x1 * f1_low + x2 * f2_low
        f2 = f1 * f1_low + f2 * f2_low
        x1 = (low_bound + R * (up_bound - low_bound)) * f1_low + x1 * f2_low
        f1 = f(x1) * f1_low + f1 * f2_low

        up_bound = x1 * f2_low + up_bound * f1_low
        x1 = x2 * f2_low + x1 * f1_low
        f1 = f2 * f2_low + f1 * f1_low
        x2 = (up_bound - R * (up_bound - low_bound)) * f2_low + x2 * f1_low
        f2 = f(x2) * f2_low + f2 * f1_low

    f1_low = (f1 < f2) * 1
    f2_low = (f1_low - 1) * (-1)
    x_opt = x1 * f1_low + x2 * f2_low

    return x_opt

##############################################################################

def find_max_x(y):

    """
    Finds the maximum x-value given a y-value subject to the GNEP constraints.
    """

    max_x = np.sqrt(abs(1 - y**2))

    return max_x

##############################################################################

def find_max_y(x):

    """
    Finds the maximum y-value given an x-value subject to the GNEP 
    constraints.
    """

    max_y = np.sqrt(abs(1 - x**2))

    return max_y

##############################################################################

def parallel_optimize_x(y, num_agents):

    """
    Calculates the upper and lower bounds for the golden_vec_search function
    and executes it for a NumPy array of x-values.

    :param y: A NumPy array of y-values
    :param num_agents: The number of x agents (ie. the size of the NumPy array
                       of x-values)

    :return: A NumPy array of the optimal x points found for all the bounds
    """

    up_bound = find_max_x(y)
    low_bound = np.zeros(num_agents)

    def funcx(x):
        return x**2 - (x * y) - x

    opt_x = golden_vec_search(low_bound, up_bound, funcx)

    return opt_x

##############################################################################

def optimize_x(y):

    """
    Optimizes a function at a single x value subject to a given y value.
    """

    def funcx(x):
        return x**2 - (x * y) - x

    bounds = (0, find_max_x(y))
    solver = minimize_scalar(funcx, bounds=bounds, method='bounded',
                             tol=None, options=None)

    opt_x = solver.x                         

    return opt_x

##############################################################################

def parallel_optimize_y(x, num_agents):

    """
    Calculates the upper and lower bounds for the golden_vec_search function
    and executes it for a NumPy array of y-values.

    :param x: A NumPy array of x-values
    :param num_agents: The number of x agents (ie. the size of the NumPy array
                       of y-values)

    :return: A NumPy array of the optimal y points found for all the bounds
    """

    up_bound = find_max_y(x)
    low_bound = np.zeros(num_agents)

    def funcy(y):
        return y**2 - (0.5 * x * y) - 2 * y

    opt_y = golden_vec_search(low_bound, up_bound, funcy)

    return opt_y

##############################################################################

def optimize_y(x):

    """
    Optimizes a function at a single y value subject to a given x value.
    """

    def funcy(y):
        return y**2 - (0.5 * x * y) - 2 * y

    bounds = (0, find_max_y(x))
    solver = minimize_scalar(funcy, bounds=bounds, method='bounded',
                             tol=None, options=None)
    
    opt_y = solver.x

    return opt_y

##############################################################################