# Our modification
This is the modification of the official TreeRep repository. We append the original README at the end of this file. Here are instructions for making it compatible for our HyperAid framework.

### Step 1.
First install Julia and PyJulia (latest version) in your virtual enviroment. Instruction can be found at the root directory of HyperAid.

### Step 2.
Install all Julia packages used in TreeRep.jl. Just modify and put the following commend at the beginning of TreeRep.jl (or NJ.jl) when you first time import them.

```
using Pkg
Pkg.add.(["LightGraphs", SparseArrays", "SimpleWeightedGraphs",...])
```
This part of code can be deleted after the sucessfull installation. 

### Step 3.
Check if you can sucessfully run the following code in TreeRep.jl
```
Main.eval('include("./TreeRep_myVer/src/TreeRep.jl")')
```
If it gives you error, please replace the relative path to absolute path. (i.e. `"./TreeRep_myVer/src/TreeRep.jl"` to `"PATH_TO_HYPERAID/TreeRep_myVer/src/TreeRep.jl"`)

### Step 4.
You should be able to sucessfully import and run all julia codes with our Python wrapper! 

# TreeRep
This is a github repository containing the code for the paper: https://arxiv.org/abs/2005.03847 (now accepted at Neurips 2020). The code is written in Julia 1.1, but should be compatible upto julia 1.5. Please cite the relavent data sources. 

Please cite using:

      @inproceedings{NEURIPS2020_093f65e0,
       author = {Sonthalia, Rishi and Gilbert, Anna},
       booktitle = {Advances in Neural Information Processing Systems},
       editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
       pages = {845--856},
       publisher = {Curran Associates, Inc.},
       title = {Tree! I am no Tree! I am a low dimensional Hyperbolic Embedding},
       url = {https://proceedings.neurips.cc/paper/2020/file/093f65e080a295f8076b1c5722a46aa2-Paper.pdf},
       volume = {33},
       year = {2020}
      }

Basic Example:

If D is the matrix with the distances. Then
  
    G2, W2  = TreeRep.metric_to_structure(D,undef,undef)
    
    G2, W2  = TreeRep.metric_to_structure_no_recursion(D,undef,undef)

will return the tree structure G2 and the Weights W2. Now if D is n by n, then W2 will be 2n by 2n (unless changed as described below). Running 

    B = W2[1:nv(G2),1:nv(G2)];
    B = sparse(B);
    B = (B .> 0) .* B;

    D2 = utilities.parallel_dp_shortest_paths(G2, B,false)[1:n,1:n];
    
Will extract the new metric on the tree. 

----------

There is also a python wrapper

-----------

TO REDUCE MEMORY USGAE: - On line 19 of TreeRep.jl change from 2n to some other fraction such as 1.2n or 1.5n or general fn for f > 1. This will siginificantly memory usage from 4n^2 to f^2n^2. However, if the learned tree doesnt fit in fn nodes (due to additional steiner nodes) this will cause a slow down of the method. 

UNLESS you are optimizing for DISTORTION DO NOT use the optimization feature for TreeRep. This is very slow and may degrade other statistics such as MAP.

The notebook in the src folder has examples for how to run the various experiments. 

Note that to use the functions in the Author helper folder you will need the code from PT and LM and PM and set up the dependencies correctly.  

--

The way the code is currently written it will not work with more than 16 threads

Trouble Shooting:

1) Memory Issues:

- Make sure the memory usage is not expected -- https://julialinearalgebra.github.io/BLASBenchmarksCPU.jl/v0.3/memory-required/
Note, the code uses FLoat64 matrices. 

- Try the above suggested change of changing 2n to 1.2n, 1.5n, 1.8n

- If slowing the code down is okay, you can try switching off multithreading and making the matrix on line 19 a sparse matrix, so spzeros(2n,2n).

- If you get a StackOverFlow error. One possible fix is to increase stack size by ulimit -s unlimited (on Ubuntu this is the command). If that doesn't work try metric_to_structure_no_recursion.

2) Run time issues

- Check if you are using multiple threads with utilities.tm()

- If the utilities.parallel_dp_shortest_paths(g,adjacency_matrix(g)) is slow this is because the D'esopo pape algorithm is usually fast, but sometimes could take exponential time (https://cp-algorithms.com/graph/desopo_pape.html). Try using one of the other shortest path algorithms instead from the LightGraphs package https://juliagraphs.org/LightGraphs.jl/latest/parallel/. 

3) Incorrect/bad tree.

- Check if the distances and the tolerance used are conflicting. That is, the tolerance should be smaller than that the distances. 

4) Other

Please open a github issue. 

    
--------

full_taxonomy.csv is from https://www.who.int/standards/classifications/classification-of-diseases

