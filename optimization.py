# Description: This file contains the implementation of the expectation maximization algorithm
# Author: Ronald Albert
# Date: June 2023
from sampling_model import galton_watson_probability
import networkx as nx
from scipy.optimize import minimize
import numpy as np

# -------------------------------------------------------------------------
# Calculate the expectation of the log-likelihood function
# -------------------------------------------------------------------------
# graph_list: A list of graphs
# alphas: A list of parameters
# theta_g: A list of parameters
# -------------------------------------------------------------------------
# Returns: The expectation of the log-likelihood function
# -------------------------------------------------------------------------
def unormalized_expectation(graph_list, alphas, theta_g):
    expected_value = 0

    # Add the last parameter to the list of parameters
    # This is done because there is a restriction on theta
    # to belong to the interval [0, 1], since it's a distribution.
    # So we assume alphas parameters to belong to the real space
    # and fit then into the [0, 1] interval with a softmax function
    alphas = np.append(alphas,1)

    # Calculate the normalization constant
    normalization_constant = np.sum(np.e**alphas)

    # Calculate the distribution theta from the parameters
    theta = (np.e**alphas)/normalization_constant 

    # Calculate the expectation 
    for graph in graph_list:
        expected_value += galton_watson_probability(graph,theta)/galton_watson_probability(graph, theta_g)

    return expected_value

# -------------------------------------------------------------------------
# Calculate the distribution that maximizes the expectation of the log-likelihood function
# -------------------------------------------------------------------------
# graph_list: A list of graphs
# theta_g: A list of parameters
# -------------------------------------------------------------------------
# Returns: The distribution that maximizes the expectation of the log-likelihood function
# -------------------------------------------------------------------------
def optimized_distribution(graph_list, theta_g):
    initial_alpha = theta_g[:-1]

    # Define the objective function
    obj_function = lambda alpha: -unormalized_expectation(graph_list, alpha, theta_g)

    # Generate a list of initial parameters
    proposed_initial_alpha = np.random.uniform(0, 1, (1000, len(initial_alpha)))

    # Find the best initial parameter for the optimization
    best_alpha = proposed_initial_alpha[np.argmin(np.apply_along_axis(obj_function, 1, proposed_initial_alpha))]

    # Minimize the objective function, starting from the best initial parameter
    # using the BFGS method for optimization
    minimization_results = minimize(obj_function, best_alpha, method='BFGS')

    # Add the last parameter to the list of parameters
    alphas = np.append(minimization_results.x, 1) 

    # Calculate the normalization constant
    normalization_constant = sum(np.e**alphas)

    # Calculate the distribution theta from the parameters
    return np.e**alphas/normalization_constant