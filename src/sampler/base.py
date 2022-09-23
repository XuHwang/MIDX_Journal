from typing import Optional
import torch
from torch import Tensor
import numpy as np

class Sampler(torch.nn.Module):
    def __init__(self, num_items, scorer_fn=None):
        super(Sampler, self).__init__()
        self.num_items = num_items-1  # remove padding
        self.scorer = scorer_fn

    def update(self, item_embs, max_iter=30):
        pass
    
    def forward(self, query, num_neg, pos_items=None):
        pass

    def compute_item_p(self, query, pos_items):
        # indeed use log(p)
        pass

class UniformSampler(Sampler):
    def __init__(self, num_items, scorer_fn=None):
        super(UniformSampler, self).__init__(num_items, scorer_fn)

    def forward(self, query: Tensor, num_neg:int, pos_items:Optional[Tensor] = None):
        with torch.no_grad():
            num_queries = np.prod(query.shape[:-1])
            shape = query.shape[:-1]

            neg_items = torch.randint(1, self.num_items + 1, size=(num_queries, num_neg), device=query.device)
            neg_items = neg_items.reshape(*shape, -1)
            neg_prob = self.compute_item_p(None, neg_items)

            if pos_items is not None:
                pos_prob = self.compute_item_p(None, pos_items)
                return pos_prob, neg_items, neg_prob
            
            else:
                return neg_items, neg_prob
    
    def compute_item_p(self, query, pos_items):
        return torch.ones_like(pos_items, dtype=torch.float)

class PopularSampler(Sampler):
    def __init__(self, pop_count, scorer_fn=None, mode=1):
        super(PopularSampler, self).__init__(pop_count.shape[0], scorer_fn)
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        
        self.register_buffer('pop_count', pop_count)
        self.register_buffer('pop_prob', pop_count / pop_count.sum())

        self.pop_prob[0] = 1
        self.register_buffer('table', torch.cumsum(self.pop_prob, dim=0))
    
    def forward(self, query, num_neg, pos_items=None):
        with torch.no_grad():
            num_queries = np.prod(query.shape[:-1])
            seeds = torch.rand(num_queries, num_neg, device=query.device)
            neg_items = torch.bucketize(seeds, self.table)
            neg_items = neg_items.reshape(*query.shape[:-1], -1)
            neg_prob = self.compute_item_p(None, neg_items)

            if pos_items is not None:
                pos_prob = self.compute_item_p(None, pos_items)
                return pos_prob, neg_items, neg_prob
            else:
                return neg_items, neg_prob

    
    def compute_item_p(self, query, pos_items):
        return torch.log(self.pop_prob[pos_items])


        
            
