# Description: Functions for the sampling model of Graphs in Galson-Watson process
# Author: Ronald Albert
# Date: June 2023
import networkx as nx
import random
import numpy as np

# -------------------------------------------------------------------------
# The operator over the auxiliary matrix that returns the number of maps between S and G
# -------------------------------------------------------------------------
# maps_matrix: The auxiliary matrix
# -------------------------------------------------------------------------
# Returns: The number of maps between S and G
# -------------------------------------------------------------------------
def _map_count_operator(maps_matrix):
    if not isinstance(maps_matrix, np.ndarray):
        return maps_matrix
    elif maps_matrix.shape[0] == 1:
        return maps_matrix.sum()
    else:
        new_maps = np.delete(maps_matrix, 0, 0)
        return sum([maps_matrix[0,j]*_map_count_operator(np.delete(new_maps, j, 1)) for j in range(new_maps.shape[1])])
    
# -------------------------------------------------------------------------
# Generate the auxiliary matrix that helps calculation in the map counting
# -------------------------------------------------------------------------
# S: Sampled graph
# G: Original graph
# S_root: Root of S
# G_root: Root of G
# -------------------------------------------------------------------------
# Returns: The auxiliary matrix
# -------------------------------------------------------------------------
def _count_maps_matrix(S, G, S_root=0, G_root=0):
    S_graph = S.copy()
    G_graph = G.copy()
    if len(S.nodes()) == 1:
        return np.array([1])
    if len(S.nodes) > len(G.nodes):
        return np.array([0])
    else:
        S_neighbors = [node for node in S_graph.neighbors(S_root)]
        G_neighbors = [node for node in G_graph.neighbors(G_root)]

        S_graph.remove_node(S_root)
        G_graph.remove_node(G_root)

        S_components = [S_graph.subgraph(c) for c in nx.weakly_connected_components(S_graph)]
        G_components = [G_graph.subgraph(c) for c in nx.weakly_connected_components(G_graph)]
        
        maps_matrix = np.empty([len(S_neighbors), len(G_neighbors)])
        for i, S_component in enumerate(S_components):
            curr_map_array = np.empty(len(G_neighbors))
            for j, G_component in enumerate(G_components):
                 
                for neighbor in S_neighbors:
                    if neighbor in S_component:
                        S_r = neighbor

                for neighbor in G_neighbors:
                    if neighbor in G_component:
                        G_r = neighbor

                maps_counted = _count_maps_matrix(S_component, G_component, S_r, G_r)
                curr_map_array[j] = _map_count_operator(maps_counted)


            maps_matrix[i,:] = curr_map_array
        
        return maps_matrix

# -------------------------------------------------------------------------
# Sampling probability of S in a graph G
# -------------------------------------------------------------------------
# S: Sampled graph
# G: Original graph
# p: Probability that a node from G is sampled into S
# -------------------------------------------------------------------------
# Returns: The probability that S is a sample subset of original graph G
# -------------------------------------------------------------------------
def sampling_probability(S, G, p, n_sampled_nodes):
    # Number of ways that S can be mapped into a subset of G
    C_gs = _map_count_operator(_count_maps_matrix(S, G)) 

    # Return the probability that S is a sampled path of G
    return C_gs * p**n_sampled_nodes * (1-p)**(len(G.nodes()) - n_sampled_nodes)
    
# -------------------------------------------------------------------------
# Sample path from a graph G with probability p
# -------------------------------------------------------------------------
# G: Graph
# p: Probability that a node from G is sampled into S
# -------------------------------------------------------------------------
# Returns: A sampled graph S of paths
# -------------------------------------------------------------------------
def sample(G, p):
    S = nx.DiGraph()
    nodes = []
    n_nodes = 0
    for node in G.nodes():
        u = random.random()
        if u < p:
            nodes.append(node)
            n_nodes += 1

    for node in nodes:
        paths = [path for path in nx.all_simple_edge_paths(G, 0, node)]
        if len(paths) > 0:
            path = paths[0]
            S.add_edges_from(path)

    return S, n_nodes

# -------------------------------------------------------------------------
# Probability of a graph G being generated from Galson-Watson process with offspring distribution
# -------------------------------------------------------------------------
# G: Graph
# offspring_distribution: A list of probabilities that a node has k offsprings
# -------------------------------------------------------------------------
# Returns: The probability of G being generated from Galson-Watson process
# -------------------------------------------------------------------------
def galton_watson_probability(G, offspring_distribution):
    out_degree = G.out_degree()
    degrees = [val for (node, val) in out_degree]

    prob = 1
    for deegre in set(degrees):
        if offspring_distribution[deegre - 1] != 0:
            prob *= offspring_distribution[deegre - 1]**degrees.count(deegre)

    return prob

# -------------------------------------------------------------------------
# Generate a Galton-Watson tree
# -------------------------------------------------------------------------
# offspring_distribution: A list of probabilities that a node has 0, 1, 2, ...
#                         children
# L: Number of levels of the tree
# -------------------------------------------------------------------------
# Returns: A Galton-Watson tree
# -------------------------------------------------------------------------
def galton_watson(offspring_distribution, L):
    G = nx.DiGraph()
    G.add_node(0)

    nodes_at_level = [[0]]

    for i in range(L - 1):
        nodes_at_level.append([])
        for v in nodes_at_level[i]:
            nodes_to_add = random.choices(range(1, len(offspring_distribution) + 1), offspring_distribution)[0]
            for _ in range(nodes_to_add):
                G.add_node(len(G.nodes()))
                G.add_edge(v, len(G.nodes())-1)
                nodes_at_level[i+1].append(len(G.nodes())-1)

    return G

# -------------------------------------------------------------------------
# Remove a tree from a graph
# -------------------------------------------------------------------------
# v: Root of the tree to be removed
# G: Graph containing the tree
# -------------------------------------------------------------------------
# Returns: The graph G with the tree rooted at v removed
# -------------------------------------------------------------------------
def remove_tree(v, G):
    if G.out_degree(v) == 0:
        G.remove_node(v)
    else:
        for u in G.copy().successors(v):
            remove_tree(u, G)
        G.remove_node(v)
    return G