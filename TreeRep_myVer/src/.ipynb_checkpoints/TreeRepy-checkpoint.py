############################################################################################################################
## This is a python wrapper to the julia package TreeRep for tree recovery based on
## Sonthalia, Rishi, and Anna C. Gilbert. 
##              "Tree! I am no Tree! I am a Low Dimensional Hyperbolic Embedding." arXiv preprint arXiv:2005.03847 (2020).
## To run this python wrapper Julia must be installed allong with all the requierd Julia packadges (see TreeRep.jl).
## The python packeage "julia" should also be installed.
## 
## A test code for this TreeRepy is availble in test_TreeRepy.py
############################################################################################################################

import julia
from julia import Main
import numpy as np
from scipy.sparse import csr_matrix,lil_matrix

Main.eval('include("/data/shared/eli/Hierachical_Clustering/HypHC/TreeRep_myVer/src/TreeRep.jl")')
Main.eval('using LightGraphs')

def TreeRep(d,rng_seed=0):
    """
    This is a python wrapper to the TreeRep algorithm written in Julia.
    
    Input:
    d - Distance matrix, assumed to be symmetric with 0 on the diagonal

    Output:
    W - Adjacenct matrix of the tree. Note that W may have rows/cols full of zeros, and they should be ignored.


    Example Code:
    d = np.array([[0,2,3],[2,0,2],[3,2,0]])
    W = TreeRep(d)
    print(W)
    """
    Main.d = d
    Main.rng_seed = rng_seed
    Main.G,Main.dist = Main.eval("TreeRep.metric_to_structure(d;rng_seed = rng_seed)")

    edges = Main.eval('collect(edges(G))')
#     W = np.zeros_like(Main.dist) # Initializing the adjacency matrix
    W = csr_matrix(Main.dist.shape)
    for edge in edges:
        src = edge.src-1 
        dst = edge.dst-1
        # We replace negative edges to 0 as in NJ.
        if Main.dist[src,dst]>0:
            W[src,dst] = Main.dist[src,dst]
        else:
            W[src,dst] = 0
        if Main.dist[dst,src]>0:
            W[dst,src] = Main.dist[dst,src]
        else:
            W[dst,src] = 0
    
    # Remove zero rows/cols
    W = W[W.getnnz(1)>0][:,W.getnnz(0)>0]
    
    return W
    
def TreeRep_no_recursion(d):
    """
    This is a python wrapper to the TreeRep algorithm written in Julia.
    
    Input:
    d - Distance matrix, assumed to be symmetric with 0 on the diagonal
    Output:
    W - Adjacenct matrix of the tree. Note that W may have rows/cols full of zeros, and they should be ignored.
    Example Code:
    d = np.array([[0,2,3],[2,0,2],[3,2,0]])
    W = TreeRep(d)
    print(W)
    """
    Main.d = d
    Main.G,Main.dist = Main.eval("TreeRep.metric_to_structure_no_recursion(d)")

    edges = Main.eval('collect(edges(G))')
    W = np.zeros_like(Main.dist) # Initializing the adjacency matrix
    for edge in edges:
        src = edge.src-1 
        dst = edge.dst-1
        W[src,dst] = Main.dist[src,dst]
        W[dst,src] = Main.dist[dst,src]

    return W
