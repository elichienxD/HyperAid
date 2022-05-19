"""Dataset loading."""

import os

import numpy as np
import torch

import ipdb

UCI_DATASETS = [
    "glass",
    "zoo",
    "iris",
    "segmentation",
    "spambase",
    "letter-recognition"
]


def load_data(dataset, normalize=True):
    """Load dataset.

    @param dataset: dataset name
    @type dataset: str
    @param normalize: whether to normalize features or not
    @type normalize: boolean
    @return: feature vectors, labels, and pairwise similarities computed with cosine similarity
    @rtype: Tuple[np.array, np.array, np.array]
    """
    if dataset in UCI_DATASETS:
        x, y = load_uci_data(dataset)
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset))
#     if normalize:
#         x = x / np.linalg.norm(x, axis=1, keepdims=True)
#     x0 = x[None, :, :]
#     x1 = x[:, None, :]
#     cos = (x0 * x1).sum(-1)
#     similarities = 0.5 * (1 + cos)
#     similarities = np.triu(similarities) + np.triu(similarities).T
#     similarities[np.diag_indices_from(similarities)] = 1.0
#     similarities[similarities > 1.0] = 1.0
    similarities = get_similarities(x)
    return x, y, similarities

def get_similarities(x,normalize=True):
    # This function is the same as the original implementation.
    # Except we use torch to perform parallel computation for speed up.
    
    # Make x as torch.tensor from numpy array
    x = torch.from_numpy(x)
    
    if normalize:
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
    x0 = x[None, :, :]
    x1 = x[:, None, :]
    cos = (x0 * x1).sum(-1)
    similarities = 0.5 * (1 + cos)
    similarities = torch.triu(similarities) + torch.triu(similarities).T
    similarities[range(similarities.shape[0]),range(similarities.shape[0])] = 1.0
    similarities[similarities > 1.0] = 1.0
    
    # Remember to make it back to numpy array
    return similarities.numpy()
    
def load_uci_data(dataset):
    """Loads data from UCI repository.

    @param dataset: UCI dataset name
    @return: feature vectors, labels
    @rtype: Tuple[np.array, np.array]
    """
    x = []
    y = []
    ids = {
        "zoo": (1, 17, -1),
        "iris": (0, 4, -1),
        "glass": (1, 10, -1),
        "segmentation": (1,19,-1),
        "spambase": (0,56,-1),
        "letter-recognition":(1,16,-1)
    }
    data_path = os.path.join(os.environ["DATAPATH"], dataset, "{}.data".format(dataset))
    classes = {}
    class_counter = 0
    start_idx, end_idx, label_idx = ids[dataset]
    with open(data_path, 'r') as f:
        for line in f:
            split_line = line.split(",")
            
            if len(split_line) >= end_idx - start_idx + 1:
                x.append([float(x) for x in split_line[start_idx:end_idx]])
                label = split_line[label_idx]
                if not label in classes:
                    classes[label] = class_counter
                    class_counter += 1
                y.append(classes[label])
    y = np.array(y, dtype=int)
    x = np.array(x, dtype=float)
    mean = x.mean(0)
    std = x.std(0)
    
    # Note that some std may = 0. In this case we just replace std=1 (as x - mean will be 0 anyway)
    std[std==0] = 1.0
    x = (x - mean) / std
    return x, y
