"""Hierarchical clustering dataset."""

import logging

import numpy as np
import torch
import torch.utils.data as data

from datasets.triples import *


class HCDataset(data.Dataset):
    """Hierarchical clustering dataset."""

    def __init__(self, features, labels, similarities, num_samples):
        """Creates Hierarchical Clustering dataset with triples.

        @param labels: ground truth labels
        @type labels: np.array of shape (n_datapoints,)
        @param similarities: pairwise similarities between datapoints
        @type similarities: np.array of shape (n_datapoints, n_datapoints)
        """
        self.features = features
        self.labels = labels
        self.similarities = similarities
        self.n_nodes = self.similarities.shape[0]
        self.triples = self.generate_triples(num_samples)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triple = self.triples[idx]
        s12 = self.similarities[triple[0], triple[1]]
        s13 = self.similarities[triple[0], triple[2]]
        s23 = self.similarities[triple[1], triple[2]]
        similarities = np.array([s12, s13, s23])
        return torch.from_numpy(triple), torch.from_numpy(similarities)

    def generate_triples(self, num_samples):
        logging.info("Generating triples.")
        if num_samples < 0:
            triples = generate_all_triples(self.n_nodes)
        else:
            triples = samples_triples(self.n_nodes, num_samples=num_samples)
        logging.info(f"Total of {triples.shape[0]} triples")
        return triples.astype("int64")


"""
Below is our new implementation
"""
    
class MetricDataset(data.Dataset):
    """Metric based hierarchical clustering dataset"""

    def __init__(self, D_metric, num_samples, labels=None):
        """Creates Metric based hierarchical Clustering dataset with distance pairs.

        @param labels: ground truth labels (if there is)
        @type labels: np.array of shape (n_datapoints,)
        @param D_metric: pairwise distances between datapoints
        @type D_metric: np.array of shape (n_datapoints, n_datapoints)
        """
        self.D_metric = D_metric
        if labels is None:
            n = D_metric.shape[0]
            labels = np.zeros(n,int) # no ground truth label, treat all labels as the same.
        self.labels = labels
        
        self.n_nodes = self.D_metric.shape[0]
        self.pairs = self.generate_pairs(num_samples)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        d12 = np.array([self.D_metric[pair[0], pair[1]]])
        return torch.from_numpy(pair), torch.from_numpy(d12)

    def generate_pairs(self, num_samples):
        logging.info("Generating pairs.")
        if num_samples < 0:
            pairs = generate_all_pairs(self.n_nodes)
        else:
            pairs = samples_pairs(self.n_nodes, num_samples=num_samples)
        logging.info(f"Total of {pairs.shape[0]} pairs")
        return pairs.astype("int64")