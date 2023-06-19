# Description: This file contains the functions for the MCMC algorithm
# Author: Ronald Albert
# Date: June 2023
import networkx as nx
import random
from sampling_model import sampling_probability, galton_watson_probability,galton_watson, remove_tree

# -------------------------------------------------------------------------
# Generating walk of length n_steps from initial_graph
# -------------------------------------------------------------------------
# initial_graph: Graph at time 0
# S: Sampled path from Galsont-Watson process
# W: Maximum out-degree of G_t
# L: Number of levels in G_t
# offspring_distribution: A list of probabilities that a node has 0, 1, 2, ...
# p: Probability of sampling a path from Galsont-Watson process
# n_steps: Number of steps in the walk
# -------------------------------------------------------------------------
# Returns: A walk of length n_steps from initial_graph
# -------------------------------------------------------------------------
def mcmc_walk(initial_graph, S, n_sampled_nodes, W, L, offspring_distribution, p, n_steps):
    G_t = initial_graph
    walk = [G_t]
    for _ in range(n_steps):
        # Generate proposal G_t+1 from G_t
        G_t_plus_1,  prob_from_G_t, prob_from_G_t_plus_1 = proposal_transition(G_t, W, L, offspring_distribution)
        # Calculate acceptance probability of G_t+1 from G_t
        accept_prob = acceptance_function(S, n_sampled_nodes, G_t, G_t_plus_1, p, offspring_distribution, prob_from_G_t, prob_from_G_t_plus_1)

        # Decide whether to accept G_t+1
        u = random.random()
        if u < accept_prob:
            G_t = G_t_plus_1
        
        walk.append(G_t)
    
    return walk


# -------------------------------------------------------------------------
# Generate proposal G_t+1 from G_t following a transition function
# -------------------------------------------------------------------------
# G_t: Graph at time t
# W: Maximum out-degree of G_t
# L: Number of levels in G_t
# offspring_distribution: A list of probabilities that a node has 0, 1, 2, ...
# -------------------------------------------------------------------------
# Returns: A proposal graph G_t+1
# -------------------------------------------------------------------------
def proposal_transition(G_t, W, L, offspring_distribution):
    # Choose an internal node v in G_t
    v = random.choice([x for x in G_t.nodes() if G_t.out_degree(x) > 0])
    d = G_t.out_degree()[v]
    l = nx.shortest_path_length(G_t, 0, v)

    # Decide whether to add or remove a tree
    if d == 1:
        action = 'add'
    elif d == W:
        action = 'remove'
    else:
        u = random.random()
        if u < 0.5:
            action = 'add'
        else:
            action = 'remove'
    
    # Add a tree to G_t
    if action == 'add':
        T_v = galton_watson(offspring_distribution, L - l - 1)
        G_t_plus_1 = nx.disjoint_union(G_t, T_v)
        G_t_plus_1.add_edge(list(G_t.nodes()).index(v), len(G_t.nodes()))

        trans_prob_from_G_t = 0.5**(d > 1) * galton_watson_probability(T_v, offspring_distribution)
        trans_prob_from_G_t_plus_1 = (0.5**(d + 1 < W)) * ((d + 1)**-1)

    # Remove a tree from G_t  
    elif action == 'remove':
        v_children = random.choice(list(G_t.successors(v)))
        G_t_plus_1 = remove_tree(v_children, G_t.copy())

        T_v = nx.DiGraph()
        T_v.add_edges_from(G_t.edges - G_t_plus_1.edges)

        trans_prob_from_G_t = (0.5**(d < W)) * (d**-1)
        trans_prob_from_G_t_plus_1 = (0.5**(d-1 > 1)) * galton_watson_probability(T_v, offspring_distribution)

    L_i = len([x for x in G_t.nodes() if G_t.out_degree(x)==0])
    L_i_plus_1 = len([x for x in G_t_plus_1.nodes() if G_t_plus_1.out_degree(x)==0])

    # Calculate the transition probabilities
    trans_prob_from_G_t = trans_prob_from_G_t/(len(G_t.nodes()) - L_i - 1)
    trans_prob_from_G_t_plus_1 = trans_prob_from_G_t_plus_1/(len(G_t_plus_1.nodes()) - L_i_plus_1 - 1)

    return G_t_plus_1, trans_prob_from_G_t, trans_prob_from_G_t_plus_1


# -------------------------------------------------------------------------
# Probbility of accepting a proposal G_t+1 from G_t
# -------------------------------------------------------------------------
# G_t: Graph at time t
# G_t_plus_1: Proposal graph at time t+1
# p: Probability of sampling a node
# offspring_distribution: A list of probabilities that a node has 0, 1, 2, ...
# trans_prob_from_G_t: Transition probability from G_t to G_t+1
# trans_prob_from_G_t_plus_1: Transition probability from G_t+1 to G_t
# -------------------------------------------------------------------------
# Returns: The probability of accepting the proposal
# -------------------------------------------------------------------------
def acceptance_function(sampled_graph, n_sampled_nodes, G_t, G_t_plus_1, p, 
                        offspring_distribution, trans_prob_from_G_t, trans_prob_from_G_t_plus_1):

    # Calculate the probability of sampling S from G_t and G_t+1
    prob_S_from_G_t_plus_1 = sampling_probability(sampled_graph, G_t_plus_1, p, n_sampled_nodes)
    prob_S_from_G_t = sampling_probability(sampled_graph, G_t, p, n_sampled_nodes)

    # Calculate the probability of sampling G_t and G_t+1 from Galton-Watson process with offspring_distribution
    prob_G_t_plus_1 = galton_watson_probability(G_t_plus_1, offspring_distribution)
    prob_G_t = galton_watson_probability(G_t, offspring_distribution)

    # Calculate the acceptance function
    acceptance = (prob_S_from_G_t_plus_1*prob_G_t_plus_1*trans_prob_from_G_t_plus_1)/(prob_S_from_G_t*prob_G_t*trans_prob_from_G_t)
    
    # Return the acceptance probability
    return min(1, acceptance)



