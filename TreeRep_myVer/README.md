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