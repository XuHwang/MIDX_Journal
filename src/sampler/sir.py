import torch
from .base import Sampler
import numpy as np
import torch.nn.functional as F

class SIR(Sampler):
    def update(self, item_embs):
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
            imp_prob = torch.softmax(score, dim=-1)
            resample_idx = torch.multinomial(imp_prob, num_samples=num_neg[1])
            neg_id = torch.gather(rand_item_idx, -1, resample_idx).view(*shape, num_neg[1])
            log_neg_prob = torch.gather(score, -1, resample_idx).view(*shape, num_neg[1])
            log_neg_prob = torch.zeros_like(log_neg_prob)
            if pos_items is not None:
                # pos_vec = torch.zeros((*pos_items.shape, self.item_vector.size(-1)), device=query.device)
                # pos_vec[pos_items > 0] = F.embedding(pos_items[pos_items>0]-1, self.item_vector)
                # log_pos_prob = self.scorer(query, pos_vec)
                log_pos_prob = torch.zeros_like(pos_items, dtype=torch.float)
                return log_pos_prob.detach(), neg_id, log_neg_prob.detach()
            else:
                return neg_id, log_neg_prob.detach()
