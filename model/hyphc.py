"""Hyperbolic hierarchical clustering model."""

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.lca import hyp_lca
from utils.linkage import nn_merge_uf_fast_np, sl_from_embeddings
from utils.poincare import project, hyp_dist

from datasets.triples import generate_all_pairs

import ipdb

class HypHC(nn.Module):
    """
    Hyperbolic embedding model for hierarchical clustering.
    """

    def __init__(self, n_nodes=1, rank=2, temperature=0.05, init_size=1e-3, max_scale=1. - 1e-3):
        super(HypHC, self).__init__()
        self.n_nodes = n_nodes
        self.temperature = temperature
#         self.scale = nn.Parameter(torch.Tensor([init_size]), requires_grad=True)
#         self.embeddings.weight.data = project(
#             self.scale * (2 * torch.rand((n_nodes, rank)) - 1.0)
#         )
        self.init_size = init_size
        self.max_scale = max_scale
        
        # Initialize embeddings in a small ball(disk) with radius in [0,init_size]
        TEMP = nn.Embedding(n_nodes, rank)
        TEMP_T = TEMP(torch.arange(n_nodes))
        TEMP_T = F.normalize(TEMP_T, p=2, dim=1)*torch.rand((n_nodes,1))*init_size
        self.embeddings = nn.Embedding.from_pretrained(TEMP_T)
        
    def anneal_temperature(self, anneal_factor):
        """

        @param anneal_factor: scalar for temperature decay
        @type anneal_factor: float
        """
        self.temperature *= anneal_factor

    def normalize_embeddings(self, embeddings):
        """Normalize leaves embeddings to have the lie on a diameter."""
        min_scale = 1e-2 #self.init_size
        max_scale = self.max_scale
        return F.normalize(embeddings, p=2, dim=1) * self.scale.clamp_min(min_scale).clamp_max(max_scale)

    def loss(self, triple_ids, similarities):
        """Computes the HypHC loss.
        Args:
            triple_ids: B x 3 tensor with triple ids
            similarities: B x 3 tensor with pairwise similarities for triples 
                          [s12, s13, s23]
        """
        e1 = self.embeddings(triple_ids[:, 0])
        e2 = self.embeddings(triple_ids[:, 1])
        e3 = self.embeddings(triple_ids[:, 2])
        e1 = self.normalize_embeddings(e1)
        e2 = self.normalize_embeddings(e2)
        e3 = self.normalize_embeddings(e3)
        d_12 = hyp_lca(e1, e2, return_coord=False)
        d_13 = hyp_lca(e1, e3, return_coord=False)
        d_23 = hyp_lca(e2, e3, return_coord=False)
        lca_norm = torch.cat([d_12, d_13, d_23], dim=-1)
        weights = torch.softmax(lca_norm / self.temperature, dim=-1)
        w_ord = torch.sum(similarities * weights, dim=-1, keepdim=True)
        total = torch.sum(similarities, dim=-1, keepdim=True) - w_ord
        return torch.mean(total)

    def decode_tree(self, fast_decoding):
        """Build a binary tree (nx graph) from leaves' embeddings. Assume points are normalized to same radius."""
        leaves_embeddings = self.normalize_embeddings(self.embeddings.weight.data)
        leaves_embeddings = project(leaves_embeddings).detach().cpu()
        sim_fn = lambda x, y: torch.sum(x * y, dim=-1)
        if fast_decoding:
            parents = nn_merge_uf_fast_np(leaves_embeddings, S=sim_fn, partition_ratio=1.2)
        else:
            parents = sl_from_embeddings(leaves_embeddings, sim_fn)
        tree = nx.DiGraph()
        for i, j in enumerate(parents[:-1]):
            tree.add_edge(j, i)
        return tree

"""
Modification below
"""
    
class MetricHypHC(nn.Module):
    """
    Metric based hyperbolic embedding model for hierarchical clustering.
    """

    def __init__(self, n_nodes=1, rank=2, temperature=0.05, init_size=1e-3, max_scale=1. - 1e-5, scaling_factor=1.0):
        super(MetricHypHC, self).__init__()
        self.n_nodes = n_nodes
        self.embeddings = nn.Embedding(n_nodes, rank,max_norm=max_scale)
#         self.scale = init_size #nn.Parameter(torch.Tensor([init_size]), requires_grad=True)
        self.embeddings.weight.data = project(
            init_size * (2 * torch.rand((n_nodes, rank)) - 1.0)
        )
        self.init_size = init_size
        self.max_scale = max_scale
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        self.all_pairs = torch.from_numpy(generate_all_pairs(n_nodes))

    def anneal_temperature(self, anneal_factor):
        """

        @param anneal_factor: scalar for temperature decay
        @type anneal_factor: float
        """
        self.temperature *= anneal_factor

    def normalize_embeddings(self, embeddings):
        """Normalize leaves embeddings to have the lie on a diameter."""
        min_scale = 1e-2 #self.init_size
        max_scale = self.max_scale
        return F.normalize(embeddings, p=2, dim=1) * self.scale.clamp_min(min_scale).clamp_max(max_scale)

    
    def loss(self, pair_ids, D_metrics, Normalization=False, p=2):
        """Computes the MetricHypHC loss.
        Args:
            triple_ids: B x 2 tensor with pair ids
            D_metrics: B x 2 tensor with pairwise distance for pairs 
        """
        e1 = self.embeddings(pair_ids[:, 0])
        e2 = self.embeddings(pair_ids[:, 1])
        if Normalization:
            # Note that in MetricHypHC loss, we don't need points lies on the diameter. It's optional.
            e1 = self.normalize_embeddings(e1)
            e2 = self.normalize_embeddings(e2)
            
        e1 = project(e1)
        e2 = project(e2)
        
        D_hyp = hyp_dist(e1,e2)
        # Note that L_inf is not supported now!
        total = torch.sum((torch.abs(D_metrics-D_hyp)/self.scaling_factor).pow(p), dim=-1, keepdim=True)
        return torch.sum(total) # Not taking **1/p for minibatch training
    
    def decode_tree(self):
        """TODO: Use TreeRep"""
        tree = None
        return tree
    
    def get_D_hyp(self,Normalization=False):
        # Note that we do not want to put things in GPU here!
        n_nodes = self.n_nodes
        all_pairs = self.all_pairs
        
        D_hyp = torch.zeros((n_nodes,n_nodes))
        e1 = self.embeddings(all_pairs[:, 0])
        e2 = self.embeddings(all_pairs[:, 1])
        if Normalization:
            # Note that in MetricHypHC loss, we don't need points lies on the diameter. It's optional.
            e1 = self.normalize_embeddings(e1)
            e2 = self.normalize_embeddings(e2)
            
        e1 = project(e1).cpu()
        e2 = project(e2).cpu()
        
        D_hyp = D_hyp.cpu()
        D_hyp[all_pairs[:,0],all_pairs[:,1]] = torch.squeeze(hyp_dist(e1,e2))
        D_hyp[all_pairs[:,1],all_pairs[:,0]] = D_hyp[all_pairs[:,0],all_pairs[:,1]]
        
        return D_hyp