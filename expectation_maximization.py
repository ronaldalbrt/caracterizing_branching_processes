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

    alphas = np.append(alphas,1)

    normalization_constant = np.sum(np.e**alphas)

    theta = (np.e**alphas)/normalization_constant 

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

    obj_function = lambda alpha: -unormalized_expectation(graph_list, alpha, theta_g)

    proposed_initial_alpha = np.random.uniform(0, 1, (1000, len(initial_alpha)))

    best_alpha = proposed_initial_alpha[np.argmin(np.apply_along_axis(obj_function, 1, proposed_initial_alpha))]

    minimization_results = minimize(obj_function, best_alpha, method='BFGS')

    alphas = np.append(minimization_results.x, 1) 

    normalization_constant = sum(np.e**alphas)

    return np.e**alphas/normalization_constant