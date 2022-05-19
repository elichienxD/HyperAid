#!/bin/bash
# source hyphc_env/bin/activate
export HHC_HOME=./
export DATAPATH="$HHC_HOME/data"                # Path where to save the data files
export SAVEPATH="$HHC_HOME/embeddings"          # Path where to save the trained models 

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python train.py --dataset zoo \
                --epochs 50 \
                --batch_size 256 \
                --learning_rate 1e-3 \
                --temperature 1e-1 \
                --eval_every 1 \
                --patience 30 \
                --optimizer RAdam \
                --anneal_every 50 \
                --anneal_factor 0.5 \
                --init_size 5e-2 \
                --num_samples 100000 \
                --seed 0 
