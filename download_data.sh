#!/bin/bash
cd ./
mkdir data
for dataset in zoo iris glass spambase; do #zoo iris glass 
  mkdir data/$dataset
  wget -P data/$dataset https://archive.ics.uci.edu/ml/machine-learning-databases/$dataset/$dataset.data
  wget -P data/$dataset https://archive.ics.uci.edu/ml/machine-learning-databases/$dataset/$dataset.names
done

# Note that image = segmentation. Also, one need to manually combined .data and .test files.
mkdir data/$dataset
wget -P data/segmentation https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data
wget -P data/segmentation https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.names
wget -P data/segmentation https://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.test