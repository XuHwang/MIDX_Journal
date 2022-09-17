import torch

class Sampler(torch.nn.Module):
    def __init__(self, num_items, scorer_fn=None):
        super(Sampler, self).__init__()
        self.num_items = num_items-1  # remove padding
        self.scorer = scorer_fn

    def update(self, item_embs, max_iter=30):
        pass

    def compute_item_p(self, query, pos_items):
        pass
