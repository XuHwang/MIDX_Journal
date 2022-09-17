import torch
from src.metric import hits, mr, mrr

from src.scorer import EuclideanScorer
from ..data import KGDataset
from src.basemodel import BaseModel

class TransE(BaseModel):

    def __init__(self, config, train_data) -> None:
        super().__init__(config, train_data)
        self.entity_embedding = torch.nn.Embedding(self.num_items, self.config['embed_dim'], 0)
        self.relation_embedding = torch.nn.Embedding(train_data.num_relations, self.config['embed_dim'], 0)
        self.score_fn = EuclideanScorer()

    def get_dataset_class():
        return KGDataset

    def construct_query(self, batch):
        r_emb = self.relation_embedding(batch['relation'])
        if 'head' in batch: # tail as target:
            h_emb = self.entity_embedding(batch['head'])
            query = h_emb + r_emb
        else:
            t_emb = self.entity_embedding(batch['tail'])
            query = t_emb - r_emb
        return query

    def encode_target(self, target):
        return self.entity_embedding(target)

    def _test_step(self, batch):
        label = batch['target']
        bs = label.size(0)
        query = self.construct_query(batch)
        scores = self.score_fn(query, self.entity_embedding.weight)
        target_score = scores[torch.arange(scores.size(0), device=scores.device), label]
        diff = scores - target_score.view(-1,1)
        metric_dict = {}
        metric_dict['mr'] = mr(diff)
        metric_dict['mrr'] = mrr(diff)
        for cutoff in self.config['cutoffs']:
            metric_dict[f'hit@{cutoff}'] = hits(diff, cutoff)
        return metric_dict, bs

    @property
    def item_vector(self):
        return self.entity_embedding.weight[1:]