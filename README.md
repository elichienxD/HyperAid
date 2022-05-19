# Our Modification
This code is developed based on the HypHC repository, which you can find their original read me at the end of this file.

## Install HyperAid
The instruction is largely based on those for HypHC.

First we create a conda enviroment.
```
conda create -n "HyperAid" python=3.8
conda activate HyperAid
```
Then install all required packages using by HypHC.
```pip install -r requirements.txt```

Note that we do not use `mst` and `unionfind` packages, albeit we include them in our code.

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

'''
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=HyperAid
'''
After that, you should be able to choose the correct kernel for Jupyter notebooks. 