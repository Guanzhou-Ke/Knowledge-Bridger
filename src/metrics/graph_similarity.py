import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def jaccard_similarity(G1, G2):
    edges_G1 = set(G1.edges())
    edges_G2 = set(G2.edges())
    
    intersection = len(edges_G1 & edges_G2)
    union = len(edges_G1 | edges_G2)
    
    return intersection / union


def graph_cosine_similarity(G1, G2):
    adj_G1 = nx.adjacency_matrix(G1).toarray()
    adj_G2 = nx.adjacency_matrix(G2).toarray()

    cos_sim = cosine_similarity(adj_G1, adj_G2)

    mean_cosine_similarity = np.mean(cos_sim)
    return mean_cosine_similarity
