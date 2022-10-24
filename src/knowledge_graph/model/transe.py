import torch
from src.metric import hits, mr, mrr

from src.scorer import EuclideanScorer
from ..data import KGDataset
from src.basemodel import BaseModel

class TransE(BaseModel):

    def add_model_specific_args(parent_parser):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group("TransE")
        parent_parser.add_argument("--embed_dim", type=int, default=500, help='embedding size')
        return parent_parser

    def __init__(self, config, train_data, epsilon=2.0) -> None:
        super().__init__(config, train_data)
        self.embedding_range = (self.config['gamma'] + epsilon) / self.config['embed_dim']

        self.entity_embedding = torch.nn.Embedding(self.num_items, self.config['embed_dim'], 0)
        self.relation_embedding = torch.nn.Embedding(train_data.num_relations, self.config['embed_dim'], 0)
        self.score_fn = EuclideanScorer(sqrt=False)
        self.sampler = self.configure_sampler()
        self._init_param()
        #TODO(@AngusHuang17): whether to try loss used in TransE (widely used in KGE papers)

    def _init_param(self):
        torch.nn.init.uniform_(
            tensor=self.entity_embedding.weight, 
            a=-self.embedding_range,
            b=self.embedding_range
        )
        torch.nn.init.uniform_(
            tensor=self.relation_embedding.weight,
            a=-self.embedding_range,
            b=self.embedding_range
        )
        self.entity_embedding.weight.data[0, :] = 0.0
        self.relation_embedding.weight.data[0, :] = 0.0

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