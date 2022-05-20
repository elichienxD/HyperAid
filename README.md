# HyperAid: Denoising in hyperbolic spaces for tree-fitting and hierarchical clustering
This is the code for our KDD 2022 paper "[HyperAid: Denoising in hyperbolic spaces for tree-fitting and hierarchical clustering](https://arxiv.org/pdf/2205.09721.pdf)". The code is developed based on both [HypHC](https://github.com/HazyResearch/HypHC) and [TreeRep](https://github.com/rsonthal/TreeRep). Please also check their repositories and readme files.

## Install HyperAid
The instruction is largely based on those for HypHC.

First we create a conda enviroment.
```
conda create -n "HyperAid" python=3.8
conda activate HyperAid
```
Then install all required packages using by HypHC.
```pip install -r requirements.txt```

Note that we do not use `mst` and `unionfind` packages for HyperAid. You may ignore them safely if you merely want to run our code.

We already included all datasets used in our experiments. However, if you want to start from scratch please run
```source download_data.sh```
Note that you need additional manual process for segmentation dataset, see comments in 'download_data.sh' for more details.

Then we install some packages that additionally used for our code
```
pip install itertools
pip install networkx
pip install newick
pip install scipy
```

Now we setup the requirements for NJ and TreeRep. First we install Julia in conda
```
conda install -c conda-forge julia
```
Please also remember the path to the julia, we will need it later. Next we install PyJulia
```
python3 -m pip install julia    # install PyJulia,  you may need `--user` after `install`
```
Please see the original [github](https://github.com/JuliaPy/pyjulia) for more instruction.

Now you should be able to run our HyperAid with decoder choice of TreeRep and NJ successfully. 

- For running the main experiments (Reproducing most part of Table 1 and 2), please check ./HyperAid.ipynb.
- For using [T-REX](http://www.trex.uqam.ca/index.php?action=home) as the decoder, please further check TREX_wrapper.ipynb.
- For using Ufit as the decoder, please git clone the official repository and put Ufit_wrapper.ipynb in their root directory.

Remark: Please remember to set up the conda enviroment as a Jupyter kernal before running Jupyter notebooks. To do so, please execute the following command.

```
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=HyperAid
```
After that, you should be able to choose the correct kernel for Jupyter notebooks. 

## Citation

If you find this code useful, please cite the following paper:

```
TBD
```

## Community Contributions
If you have developed or implemented any tree-metric or ultra-metric methods and you want to combine with HyperAid, please either submit a pull request or contact me (ichien3@illinois.edu). Also, if you managed to make a python version of NJ (even with advanced clean up steps as in TREX or scalable version), please let me know! I would really appriciate such contributions!

## Issues
If you find any issues about our code, please open an issue **and** email me (ichien3@illinois.edu) in case I do not receive the notification correctly. I will try to address them as soon as possible :)
