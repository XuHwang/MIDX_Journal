import torch
import numpy as np
import torch.nn.functional as F

from .base import UniformSampler
from typing import Union
from .midx import construct_index


class LSHSampler(UniformSampler):

    def __init__(
            self, 
            num_items: int,
            n_dims: int,
            n_bits: int = 4,
            n_table: int = 16, 
            device: Union[str, torch.device]="cuda", 
            scorer_fn=None
        ):
        """
        LSH-based negative sampler proposed in 
        "A New Unbiased and Efficient Class of LSH-Based Samplers and Estimators for Partition Function Computation in Log-Linear Models".

        Args:
            num_items (int): number of items
            n_dims (int): dimension of item embedding vectors
            n_bits (int): number of hash functions in each hash table, i.e., K in the paper
            n_table (int): number of hash tables, i.e., L in the paper
            device (str or torch.device): device to save the hash functions, must be consistent with item embedding vectors
        """
        super().__init__(num_items, scorer_fn)
        self.n_dims = n_dims
        self.n_bits = n_bits
        self.n_table = n_table
        self.device = device
        self.weight_vectors = self._generate_random_vectors(self.n_dims, self.n_bits, self.n_table) # DxKxL, normalized
        K_base_vec = torch.from_numpy(1 << np.arange(n_bits - 1, -1, -1)).type(torch.float).to(self.device)  # [K]
        self.K_base_vec = torch.nn.parameter.Parameter(K_base_vec, requires_grad=False)
        self.indptr, self.indices = None, None
        self.item_embs = None

    @torch.no_grad()
    def update(self, item_embs: torch.Tensor):
        norm_item_embs = item_embs / (torch.norm(item_embs, dim=1, keepdim=True) + 1e-10)
        y = torch.matmul(norm_item_embs, self.weight_vectors.view(self.n_dims, -1)).view(item_embs.size(0), self.n_bits, -1)    # NxKxL
        y = (y > 0).type(torch.float)
        code = torch.matmul(y.transpose(1,2), self.K_base_vec)   # N, L
        self.item_embs = item_embs.clone().detach() 
        self.indices, self.indptr = self._construct_inverted_index(code)   # indptr: Lx(K+1); indices: LxN

    @torch.no_grad()
    def forward(self, query, num_neg, pos_items=None):
        """
        Sample negative items and calculate sampling probablity for correction.

        Args:
            query (torch.Tensor): shape of (B,D), query embedding. 
            num_neg (int): number of negative items to be sampled
            pos_items (torch.Tensor): shape of (B), positive item indexes.

        Returns:
            log_pos_prob (torch.Tensor): log sampling probability of positive items
            neg_id (torch.Tensor): sampled negative item indexes
            log_neg_prob (torch.Tensor): log sampling probability of negative items
        """
        # get hash code
        query_shape = query.shape
        query = query.reshape(-1, query.size(-1))
        norm_query = query / (torch.norm(query, dim=-1, keepdim=True) + 1e-10)
        y = torch.matmul(norm_query, self.weight_vectors.view(self.n_dims, -1)).view(query.size(0), self.n_bits, -1)    # BxKxL
        y = (y > 0).type(torch.float)
        code = torch.matmul(y.transpose(1,2), self.K_base_vec).transpose(0, 1).type(torch.long)   # LxB
        start_idx = torch.gather(self.indptr, dim=1, index=code)    # LxB
        end_idx = torch.gather(self.indptr, dim=1, index=code+1)    # LxB
        num_candidates = (end_idx - start_idx)
        len_item = num_candidates.sum(dim=0) # B

        # for empty candidates, use uniform sampling
        empty_flag = (len_item == 0)
        if empty_flag.any():
            neg_id_empty, log_neg_prob_empty = super().forward(query[empty_flag], num_neg, pos_items=None)

        cum_len = num_candidates.cumsum(dim=0).T.contiguous()
        rand_idx = torch.floor(torch.rand((query.size(0), num_neg), device=query.device) * len_item.view(-1, 1)).type(torch.long)   # B x neg
        rand_idx[rand_idx==len_item.view(-1,1)] = rand_idx[rand_idx==len_item.view(-1,1)] - 1   # in case of numerically unstable
        table_id = torch.searchsorted(cum_len, rand_idx, right=True)    # B x neg
        _table_id = table_id - 1
        flag = _table_id < 0
        _table_id[flag] = 0
        offset = torch.gather(cum_len, dim=1, index=_table_id)
        offset[flag] = 0
        offset = rand_idx - offset
        indices = torch.gather(start_idx.transpose(0,1), dim=1, index=table_id) + offset    # B x neg
        item_id = self.indices[table_id, indices]   # B x neg

        # cal probablity
        sampling_prob = 1.0 / (len_item) # B
        sampled_item_emb = F.embedding(item_id, self.item_embs, padding_idx=0)  # B x neg x D
        cosine_theta = F.cosine_similarity(query.view(query.size(0), 1, query.size(1)), sampled_item_emb, dim=-1)   # B x neg
        theta = torch.acos(cosine_theta)
        collision_p = 1 - theta / torch.pi
        weight = (1 - (1 - collision_p ** self.n_bits) ** self.n_table)
        neg_prob = sampling_prob.view(-1, 1) * weight
        neg_id = item_id + 1    # item_id denotes item index without padding

        eps = 1e-12
        log_neg_prob = torch.log(neg_prob + eps)
        if empty_flag.any():
            neg_id[empty_flag] = neg_id_empty
            log_neg_prob[empty_flag] = log_neg_prob_empty.type_as(log_neg_prob)
        
        neg_id = neg_id.view(query_shape[:-1] + (num_neg,))
        log_neg_prob = log_neg_prob.view(query_shape[:-1] + (num_neg,))
        if pos_items is not None:   # get correction for positive items
            # return torch.zeros_like(pos_items), neg_id, log_neg_prob    
            return torch.zeros_like(pos_items), neg_id, torch.zeros_like(neg_id)    
        else:
            # return item_id + 1, torch.log(neg_prob)
            return item_id + 1, torch.zeros_like(neg_prob)


    def _generate_random_vectors(self, n_dims, n_hash, n_table):
        random_vectors = torch.randn(n_dims, n_hash, n_table)    # DxKxL
        norm_random_vectors = random_vectors / (torch.norm(random_vectors, dim=0, keepdim=True))
        return torch.nn.Parameter(norm_random_vectors.to(self.device), requires_grad=False)


    def _construct_inverted_index(self, code):
        """
        Construct inverted index for each hash table with a csr sparse data structure.

        Args:
            idx (np.ndarray): NxL, where N is the number of data points, L is the number of tables. Each entry ranges from 0 to K-1. 
                              It indicates each data point's hash code.
        """
        table_indptr = []
        table_indices = []
        for i in range(self.n_table):
            indptr, indices = construct_index(code[:, i], 2 ** self.n_bits)
            table_indptr.append(indptr)
            table_indices.append(indices)
        table_indptr = torch.stack(table_indptr)
        table_indices = torch.stack(table_indices)

        return table_indptr, table_indices