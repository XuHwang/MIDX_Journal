from xml.dom import NotSupportedErr
import torch
import torch.nn as nn
from ..data import ExtremeClassDataset
from src.basemodel import BaseModel
from src.metric import precision, ndcg

class EcFc(BaseModel):
    def __init__(self, config:dict, train_data:ExtremeClassDataset) -> None:
        super().__init__(config, train_data)
        
        self.class_embedding = torch.nn.Embedding(self.num_items, self.config['embed_dim'], padding_idx=0)

        if isinstance(self.config['hidden_states'], int):
            dims = [self.config['hidden_states'], self.config['embed_dim']]
        elif isinstance(self.config['hidden_states'], list):
            dims = self.config['hidden_states'].copy()
            dims.append(self.config['embed_dim'])
        elif self.config['hidden_states'] is None:
            dims = [self.config['embed_dim']]
        else:
            raise NotSupportedErr('Unsupported type hidden states')

        self.feat_embedding = torch.nn.Embedding(self.num_feat + 1, dims[0], padding_idx=0)
        
        if len(dims) > 1:
            self.mlp_layers = nn.Sequential()
        
            for idx in range(1, len(dims)):
                self.mlp_layers.add_module('fc_{}'.format(idx), nn.Linear(dims[idx-1], dims[idx]))

                if idx < (len(dims) - 1):
                    self.mlp_layers.add_module('act_{}'.format(idx), nn.ReLU())


    def get_dataset_class():
        return ExtremeClassDataset
    
    def construct_query(self, batch):
        res = (self.feat_embedding(batch['feat_col']) * batch['feat_value'].unsqueeze(-1)).sum(1)
        if hasattr(self, "mlp_layers"):
            res = self.mlp_layers(res)
        return res
    
    def encode_target(self, target):
        return self.class_embedding(target)

    
    def _test_step(self, batch):
        topk = self.config['topk']
        cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], list) else [self.config['cutoff']]
        bs = batch['target'].size(0)
        query = self.construct_query(batch)
        scores = self.score_fn(query, self.item_vector)
        topk_scores, topk_items = torch.topk(input=scores, k=topk, dim=-1)
        topk_items = topk_items + 1
        target, _ = batch['target'].sort()
        idx_ = torch.searchsorted(target, topk_items)
        idx_[idx_ == target.size(1)] = target.size(1) - 1
        pred = torch.gather(target, 1, idx_) == topk_items
        target = batch['target'].gt(0).long()
        metric_dict = {}
        for cutoff in cutoffs:
            metric_dict[f'prec@{cutoff}'] = precision(pred, target, cutoff)
            metric_dict[f'ndcg@{cutoff}'] = ndcg(pred, target, cutoff)
        return metric_dict, bs


    @property
    def item_vector(self):
        return self.class_embedding.weight[1:]