import torch
from src.metric import hits, mr, mrr

from src.scorer import EuclideanScorer
from ..data import KGDataset
from src.basemodel import BaseModel

class HousE(BaseModel):

    def add_model_specific_args(parent_parser):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group("HousE")
        parent_parser.add_argument("--embed_dim", type=int, default=500, help='embedding dimension')
        parent_parser.add_argument("--house_dim", type=int, default=2, help='house dimension')
        parent_parser.add_argument("--house_num", type=int, default=2, help='house number')
        return parent_parser

    def __init__(self, config, train_data, epsilon=2.0) -> None:
        super().__init__(config, train_data)
        self.num_relations = train_data.num_relations
        self.hidden_dim, self.house_num, self.house_dim \
             = self.config['embed_dim'], self.config['house_num'], self.config['house_dim']
        
        self.embedding_range = \
            torch.Tensor([(self.config['gamma'] + epsilon) / (self.hidden_dim * (self.house_dim ** 0.5))]),
        self.entity_embedding = torch.nn.Parameter(torch.zeros(self.num_items, self.hidden_dim, self.house_dim))
        
        self.relation_embedding = torch.nn.Parameter(torch.zeros(train_data.num_relations, self.hidden_dim, self.house_dim*self.house_num)) # num_r x d x k*m
        
        self.score_fn = EuclideanScorer()
        self.sampler = self.configure_sampler()
        self._init_param()
        #TODO(@AngusHuang17): whether to try loss used in RotatE (widely used in KGE papers)

    def _init_param(self):
        torch.nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range,
            b=self.embedding_range
        )
        torch.nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range,
            b=self.embedding_range
        )
        self.entity_embedding.data[0, :] = 0.0
        self.relation_embedding.data[0, :] = 0.0

    def norm_embedding(self):
        entity_embedding = self.entity_embedding
        r_list = torch.chunk(self.relation_embedding, self.house_num, 2)
        normed_r_list = []
        for i in range(self.house_num):
            r_i = torch.nn.functional.normalize(r_list[i], dim=2, p=2)
            normed_r_list.append(r_i)
        r = torch.cat(normed_r_list, dim=2)
        # self.k_head = self.k_dir_head * torch.abs(self.k_scale_head)
        # self.k_head[self.k_head > self.thred] = self.thred
        # self.k_tail = self.k_dir_tail * torch.abs(self.k_scale_tail)
        # self.k_tail[self.k_tail > self.thred] = self.thred
        return entity_embedding, r

    def get_dataset_class():
        return KGDataset

    def construct_query(self, batch):
        entity_embedding, r = self.norm_embedding()
        r_emb = torch.index_select(r, dim=0, index=batch['relation'])
        r_list = torch.chunk(r_emb, self.house_num, dim=-1)
        
        if 'head' in batch: # tail as target:
            head = torch.index_select(entity_embedding, dim=0, index=batch['head'])
            for i in range(self.house_num):
                j = self.house_num - 1 - i
                head = head - 2 * (r_list[j] * head).sum(dim=-1, keepdim=True) * r_list[j]
            return head.view(head.size(0), -1) # [B, D*k]
        else:
            # k_head = torch.index_select(self.k_head, dim=0, index=batch['head'])
            tail = torch.index_select(entity_embedding, dim=0, index=batch['tail'])
            for i in range(self.house_num):
                tail = tail - 2 * (r_list[i] * tail).sum(dim=-1, keepdim=True) * r_list[i]
            return tail.view(tail.size(0), -1) # [B, D, k]


    def encode_target(self, target):
        target_emb = torch.index_select(self.entity_embedding, dim=0, index=target)
        return target_emb.view(target_emb.size(0), -1)


    def _test_step(self, batch):
        label = batch['target']
        bs = label.size(0)
        query = self.construct_query(batch)
        all_emb = self.entity_embedding
        scores = self.score_fn(query, all_emb.view(all_emb.size(0), -1))
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
        emb = self.entity_embedding[1:]
        return emb.view(emb.size(0), -1) # [N, D, k]