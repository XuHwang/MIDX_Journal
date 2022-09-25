import torch
from .base import Sampler
import numpy as np
import torch.nn.functional as F

class DynamicSampler(Sampler):
    def update(self, item_embs):
        # this is a pointer, the value will update automatically
        self.item_vector = item_embs

    def forward(self, query, num_neg, pos_items=None):
        if isinstance(num_neg, int):
            num_neg = [2*num_neg, num_neg]
        num_queries = np.prod(query.shape[:-1])
        shape = query.shape[:-1]
        with torch.no_grad():
            rand_item_idx = torch.randint(1, self.num_items+1, size=(num_queries, num_neg[0]), device=query.device)
            rand_item_emb = F.embedding(rand_item_idx-1, self.item_vector)
            score = self.scorer(query, rand_item_emb.view(*shape, num_neg[0], -1)).view(-1, num_neg[0])
            _, top_idx = torch.topk(score, k=num_neg[1], dim=-1)
            neg_id = torch.gather(rand_item_idx, dim=-1, index=top_idx).view(*shape, num_neg[1])
            log_neg_prob = torch.zeros_like(neg_id)
            if pos_items is not None:
                log_pos_prob = torch.zeros_like(pos_items)
                return log_pos_prob.detach(), neg_id, log_neg_prob.detach()        
            else:
                return neg_id, log_neg_prob.detach() 
