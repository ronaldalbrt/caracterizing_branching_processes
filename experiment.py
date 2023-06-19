# Definition of the experiment.
# Author: Ronald Albert
# Date: June 2023
from sampling_model import galton_watson, sample
from graph_mcmc import mcmc_walk
from optimization import optimized_distribution
import numpy as np

# -------------------------------------------------------------------------
# Calculate the Kullback-Leibler divergence between two distributions
# -------------------------------------------------------------------------
# p: Distribution parameter p
# q: Distribution parameter q
# -------------------------------------------------------------------------
# Returns: The Kullback-Leibler divergence between p and q
# -------------------------------------------------------------------------
def kl_divergence(p, q):
    return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)))


# -------------------------------------------------------------------------
# Run the experiment
# -------------------------------------------------------------------------
# offspring_distribution: The offspring distribution
# dist_g: Distribution for the MCMC algorithm
# W: Maximum value of the offspring distribution
# L: Maximum number of levels in the tree
# p: Probability of sampling a path from the Galton-Watson process
# n_steps: Number of steps in the walk
# -------------------------------------------------------------------------
# Returns: The estimated distribution
# -------------------------------------------------------------------------
def experiment(offspring_distribution, dist_g, W, L, p, n_steps):
    G = galton_watson(offspring_distribution, L)

    S, n_nodes = sample(G, p)

    sampled_graphs = mcmc_walk(G, S, n_nodes, W, L, dist_g, p, n_steps)

    theta = optimized_distribution(sampled_graphs, dist_g)

    return kl_divergence(theta, offspring_distribution), theta

    
