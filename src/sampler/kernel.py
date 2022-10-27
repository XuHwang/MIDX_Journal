import torch, math
from .base import Sampler
from ..scorer import InnerProductScorer
import torch.nn.functional as F
import numpy as np

class KernelSampler(Sampler):
    """
        Adaptive Sampled Softmax with Kernel Based Sampling
        Sampled Softmax with Random Fourier Features
    """
    def __init__(self, num_items, scorer_fn=None):
        assert isinstance(scorer_fn, InnerProductScorer)
        super().__init__(num_items, scorer_fn)
    
    def update(self, item_embs, max_iter=30):
        self.item_vec = item_embs # without padding values
    
    def get_logits(self, query):
        pass
    
    def forward(self, query, num_neg, pos_items=None):
        with torch.no_grad():
            logits = self.get_logits(query)
            logits = logits.reshape(-1, self.item_vec.shape[0])
            
            neg_items = torch.multinomial(logits, num_samples=num_neg, replacement=True)

            neg_prob = torch.log( torch.gather(logits, -1, neg_items) * num_neg) - torch.reshape(torch.log(logits.sum(-1)), [*logits.shape[:-1], 1])

        if pos_items is not None:
            pos_prob = torch.zeros_like(pos_items)
            return pos_prob, neg_items.reshape(*query.shape[:-1], -1) + 1, neg_prob.reshape(*query.shape[:-1], -1)
        
        else:
            return  neg_items.reshape(*query.shape[:-1], -1) + 1, neg_prob.reshape(*query.shape[:-1], -1)

    def compute_item_p(self, query, pos_items):
        return super().compute_item_p(query, pos_items)

class SphereSampler(KernelSampler):
    def __init__(self, num_items, scorer_fn=None):
        super().__init__(num_items, scorer_fn)
        
    def get_logits(self, query):
        logits = self.scorer(query, self.item_vec)
        return 100 * logits ** 2 + 1
    

class RFFSampler(KernelSampler):
    """
        Refer to the paper: Sampled Softmax with Random Fourier Features

        
        More cases can refer to : 
        [1] https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/RandomFourierFeatures
        [2] http://random-walks.org/content/misc/rff/rff.html
    """
    def __init__(self, num_items, scorer_fn=None):
        super().__init__(num_items, scorer_fn)
        self.temperature = 5 # \tau denotes the temp for softmax function #TODO: implememt into softmax loss
        # According to the paper, "the typical choice ranges from 5 to 30."
        self.nu = 4.0  # nu denotes \nu
        # According to the paper, the value of \nu should be smaller than temperature (\tau)
        # In experiments, \nu = 4 achieves the best performance
        self.num_random_features = 32 # config parameter
    
    @staticmethod
    def kernel_vec(item_vec, temp, num_random_features):
        func = InnerProductScorer()
        item_vec = F.normalize(item_vec, dim=-1) # TODO : ensure || c || = 1
        shape = []
        for i in range(item_vec.dim()-1):
            shape.append(item_vec.shape[i])
        shape.append(num_random_features)
        shape.append(item_vec.shape[-1])

        sampled_w = torch.normal(0, math.sqrt(1/temp), size=tuple(shape), device=item_vec.device)
        _scores = func(item_vec, sampled_w)

        return 1/math.sqrt(num_random_features) * torch.cat([torch.cos(_scores), torch.sin(_scores)], dim=-1)
    
    def update(self, item_embs, max_iter=30):
        self.item_vec = RFFSampler.kernel_vec(item_embs, self.nu, self.num_random_features)
    
    def get_logits(self, query):
        kernel_query_vec = RFFSampler.kernel_vec(query, self.nu, self.num_random_features)
        return self.scorer(kernel_query_vec, self.item_vec)
    


