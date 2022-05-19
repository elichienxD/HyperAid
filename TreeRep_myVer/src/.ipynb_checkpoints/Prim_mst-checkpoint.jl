module Prim_mst

# import Pkg
# Pkg.add("Graphs")
# Pkg.add("SimpleWeightedGraphs")

#Include
using Graphs, SimpleWeightedGraphs

"""
Call prim_mst from Graphs. 
"""
function Prim_mst!(D::Matrix{Float64})

    # Construct undirected weighted graph from adjacency matrix D.
    g = SimpleWeightedGraph(D)
    # Run prim_mst
    edgelist = prim_mst(g)
    # Turn edgelist into an array
    output = [[src(e), dst(e)] for e in edgelist]

    return output
end



end
