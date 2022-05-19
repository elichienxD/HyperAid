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



=================================================================================================================
# Hyperbolic Hierarchical Clustering (HypHC)

This code is the official PyTorch implementation of the NeurIPS 2020 paper: 
> **From Trees to Continuous Embeddings and Back: Hyperbolic Hierarchical Clustering**\
> Ines Chami, Albert Gu, Vaggos Chatziafratis and Christopher Ré\
> Stanford University\
> Paper: https://arxiv.org/abs/2010.00402

<p align="center">
  <img width="400" height="400" src="https://github.com/HazyResearch/HypHC/blob/master/HypHC.gif">
</p>

> **Abstract.** Similarity-based Hierarchical Clustering (HC) is a classical unsupervised machine learning algorithm that has traditionally been solved with heuristic algorithms like Average-Linkage. Recently, Dasgupta reframed HC as a discrete optimization problem by introducing a global cost function measuring the quality of a given tree. In this work, we provide the first continuous relaxation of Dasgupta's discrete optimization problem with provable quality guarantees. The key idea of our method, HypHC, is showing a direct correspondence from discrete trees to continuous representations (via the hyperbolic embeddings of their leaf nodes) and back (via a decoding algorithm that maps leaf embeddings to a dendrogram), allowing us to search the space of discrete binary trees with continuous optimization. Building on analogies between trees and hyperbolic space, we derive a continuous analogue for the notion of lowest common ancestor, which leads to a continuous relaxation of Dasgupta's discrete objective. We can show that after decoding, the global minimizer of our continuous relaxation yields a discrete tree with a (1+epsilon)-factor approximation for Dasgupta's optimal tree, where epsilon can be made arbitrarily small and controls optimization challenges. We experimentally evaluate HypHC on a variety of HC benchmarks and find that even approximate solutions found with gradient descent have superior clustering quality than agglomerative heuristics or other gradient based algorithms. Finally, we highlight the flexibility of HypHC using end-to-end training in a downstream classification task.


## Installation

This code has been tested with python3.7. First, create a virtual environment (or conda environment) and install the dependencies:

```python3 -m venv hyphc_env```

```source hyphc_env/bin/activate```

```pip install -r requirements.txt``` 

Then install the ```mst``` and ```unionfind``` packages which are used to decode embeddings into trees and compute the discrete Dasgupta Cost efficiently: 

```cd mst; python setup.py build_ext --inplace```

```cd unionfind; python setup.py build_ext --inplace```

## Datasets

```source download_data.sh```

This will download the zoo, iris and glass datasets from the UCI machine learning repository. Please refer to the paper for the download links of the other datasets used in the paper. 

## Code Usage

### Train script

To use the code, first set environment variables in each shell session:

```source set_env.sh```

To train the HypHC mode, use the train script:
```
python train.py
    optional arguments:
      -h, --help            show this help message and exit
      --seed SEED
      --epochs EPOCHS
      --batch_size BATCH_SIZE
      --learning_rate LEARNING_RATE
      --eval_every EVAL_EVERY
      --patience PATIENCE
      --optimizer OPTIMIZER
      --save SAVE
      --fast_decoding FAST_DECODING
      --num_samples NUM_SAMPLES
      --dtype DTYPE
      --rank RANK
      --temperature TEMPERATURE
      --init_size INIT_SIZE
      --anneal_every ANNEAL_EVERY
      --anneal_factor ANNEAL_FACTOR
      --max_scale MAX_SCALE
      --dataset DATASET
``` 

### Examples

We provide examples of training commands for the zoo, iris and glass datasets. For instance, to train HypHC on zoo, run: 

```source examples/run_zoo.sh``` 

This will create an `embedding` directory and save training logs, embeddings and the configuration parameters in a `embedding/zoo/[unique_id]` where the unique id is based on the configuration parameters used to train the model.   

## Citation

If you find this code useful, please cite the following paper:

```
@inproceedings{NEURIPS2020_ac10ec1a,
 author = {Chami, Ines and Gu, Albert and Chatziafratis, Vaggos and R\'{e}, Christopher},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {15065--15076},
 publisher = {Curran Associates, Inc.},
 title = {From Trees to Continuous Embeddings and Back: Hyperbolic Hierarchical Clustering},
 url = {https://proceedings.neurips.cc/paper/2020/file/ac10ec1ace51b2d973cd87973a98d3ab-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
