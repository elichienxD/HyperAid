############################################################################################################################
## This is a python wrapper to the julia package NJ for tree recovery used in the paper 
## Sonthalia, Rishi, and Anna C. Gilbert. 
##              "Tree! I am no Tree! I am a Low Dimensional Hyperbolic Embedding." arXiv preprint arXiv:2005.03847 (2020).
## To run this python wrapper Julia must be installed allong with all the requierd Julia packadges (see NJ.jl).
## The python packeage "julia" should also be installed.
## 
## The julia implementation is from Rishi and the author (me) only write the python wrapper.
############################################################################################################################
import ipdb
import julia
from julia import Main
import numpy as np
from scipy.sparse import csr_matrix,lil_matrix

Main.eval('include("/data/shared/eli/Hierachical_Clustering/HypHC/TreeRep_myVer/src/NJ.jl")')
Main.eval('using LightGraphs')

def NJ(d,rng_seed=0):
    """
    This is a python wrapper to the NJ algorithm written in Julia by Rishi.
    
    Input:
    d - Distance matrix, assumed to be symmetric with 0 on the diagonal

    Output:
    W - Adjacenct matrix of the tree. Note that W may have rows/cols full of zeros, and they should be ignored.
    """
    Main.d = d
    Main.rng_seed = rng_seed
    Main.G = Main.eval("NJ.nj!(d)")
    
    edges = Main.eval('collect(G.edge)')

#     W = np.zeros((2*d.shape[0],2*d.shape[0])) # Initializing the adjacency matrix
    W = lil_matrix((2*d.shape[0],2*d.shape[0])) # Note that we should use csr matrix, as 0 edge weight != no edge.
    for edge in edges:
#         ipdb.set_trace()
        src = edge.node[0].number - 1
        dst = edge.node[1].number - 1
        W[src,dst] = edge.length
        W[dst,src] = edge.length
    
    W = csr_matrix(W)
    # Remove zero rows/cols
    W = W[W.getnnz(1)>0][:,W.getnnz(0)>0]
    
    
    return W
