import torch
from src.loss_func import FullSoftmax, RotatELoss, SampledSoftmax, TransELoss, PRISLoss
from src.metric import hits, mr, mrr

from src.scorer import EuclideanScorer, InnerProductScorer
from ..data import KGDataset
from src.basemodel import BaseModel

class TransE(BaseModel):

    def add_model_specific_args(parent_parser):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group("TransE")
        parent_parser.add_argument("--embed_dim", type=int, default=500, help='embedding size')
        parent_parser.add_argument("--gamma", type=float, default=2, help='margin in loss')
        parent_parser.add_argument("--loss", type=str, default='ssl', help='loss function')
        parent_parser.add_argument("--weight", type=str, default='True', help='whether to use weight in loss')
        parent_parser.add_argument("--norm", action='store_true', default=False, help='whether to norm entity embedding')
        return parent_parser

    def __init__(self, config, train_data, epsilon=2.0) -> None:
        super().__init__(config, train_data)
        # self.embedding_range = (self.config['gamma'] + epsilon) / self.config['embed_dim']
        self.embedding_range = 0.01
        self.entity_embedding = torch.nn.Embedding(self.num_items, self.config['embed_dim'], 0)
        self.relation_embedding = torch.nn.Embedding(train_data.num_relations, self.config['embed_dim'], 0)
        norm = self.config['norm']
        self.score_fn = InnerProductScorer() if norm else EuclideanScorer()
        self.sampler = self.configure_sampler()
        self.loss_fn = self.configure_loss(self.config['loss'])
        self._init_param()
        #TODO(@AngusHuang17): whether to try loss used in TransE (widely used in KGE papers)

    def configure_loss(self, loss='ssl'):
        loss = loss.lower()
        if self.sampler is not None:
            if loss == 'ssl':
                return SampledSoftmax()
            elif loss == 'tsl':
                return TransELoss(self.config['gamma'], weight=(self.config['weight']=='True'))
            elif loss == 'rtl':
                return RotatELoss(self.config['gamma'], weight=(self.config['weight']=='True'))
            elif loss == 'pris':
                return PRISLoss(self.config['gamma'], weight=(self.config['weight']=='True'))
            else:
                return ValueError("No such loss. Only support [ssl/tsl/rtl]")
        else:
            return FullSoftmax()

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
        norm = self.config['norm']
        r_emb = self.relation_embedding(batch['relation'])
        if 'head' in batch: # tail as target:
            h_emb = self.entity_embedding(batch['head'])
            if norm:
                h_emb = h_emb / torch.norm(h_emb, p=2, dim=-1, keepdim=True)
                query = 2 * (h_emb + r_emb)
            else:
                query = (h_emb + r_emb)
        else:
            t_emb = self.entity_embedding(batch['tail'])
            if norm:
                t_emb = t_emb / torch.norm(t_emb, p=2, dim=-1, keepdim=True)
                query = 2 * (t_emb - r_emb)
            else:
                query = (t_emb - r_emb)
        return query

    def encode_target(self, target):
        emb = self.entity_embedding(target)
        if self.config['norm']:
            emb = emb / torch.norm(emb, p=2, dim=-1, keepdim=True)
        return emb

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
        embs = self.entity_embedding.weight[1:]
        if self.config['norm']:
            embs = embs / torch.norm(embs, p=2, dim=-1, keepdim=True)
        return embs