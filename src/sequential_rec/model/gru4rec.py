import torch
from src.basemodel import BaseModel
from src.metric import ndcg, recall
from ..data import SeqDataset

class GRU4Rec(BaseModel):

    def add_model_specific_args(parent_parser):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('GRU4Rec')
        parent_parser.add_argument("--embed_dim", type=int, default=64, help='embededding dimension')
        parent_parser.add_argument("--hidden_size", type=int, default=128)
        parent_parser.add_argument("--layer_num", type=int, default=2)
        parent_parser.add_argument("--dropout_rate", type=float, default=0.5, help='dropout rate')

        return parent_parser

    def __init__(self, config, train_data) -> None:
        super().__init__(config, train_data)
        self.embed_dim = self.config['embed_dim']
        self.hidden_size = self.config['hidden_size']
        self.num_layers = self.config['layer_num']
        self.dropout_rate = self.config['dropout_rate']
        self.emb_dropout = torch.nn.Dropout(self.dropout_rate)
        self.GRU = torch.nn.GRU(
            input_size = self.embed_dim,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            bias = False,
            batch_first = True,
            bidirectional = False
        )
        self.dense = torch.nn.Linear(self.hidden_size, self.embed_dim)

        self.fiid = train_data.fiid
        self.frating = train_data.frating
        self.item_encoder = self._get_item_encoder(train_data)

    def get_dataset_class():
        return SeqDataset
    
    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    def construct_query(self, batch):
        user_hist = batch['in_' + self.fiid]
        emb_hist = self.item_encoder(user_hist)
        emb_hist_dropout = self.emb_dropout(emb_hist)   # B x L x H_in
        gru_vec, _ = self.GRU(emb_hist_dropout)    # B x L x H_out
        query = self.dense(gru_vec)
        gather_index = (batch['seqlen']-1).view(-1, 1, 1).expand(-1, -1, query.shape[-1]) # B x 1 x H_out
        query_output = query.gather(dim=1, index=gather_index).squeeze(1)  # B x H_out
        return emb_hist, query_output

    def topk(self, query, k, user_h):
        more = user_h.size(1) if user_h is not None else 0
        score, topk_items = torch.topk(self.score_fn(query, self.item_vector), k + more)
        if user_h is not None:
            topk_items += 1
            existing, _ = user_h.sort()
            idx_ = torch.searchsorted(existing, topk_items)
            idx_[idx_ == existing.size(1)] = existing.size(1) - 1
            score[torch.gather(existing, 1, idx_) == topk_items] = -float('inf')
            score1, idx = score.topk(k)
            return score1, torch.gather(topk_items, 1, idx)
        else:
            return score, topk_items

    def _test_step(self, batch):
        topk = self.config['topk']
        cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], list) else [self.config['cutoff']]
        bs = batch[self.frating].size(0)
        with torch.no_grad():
            query = self.construct_query(batch)
            scores = self.score_fn(query, self.item_vector)
        topk_scores, topk_items = self.topk(query, topk, batch['user_hist'])        
        pred = batch[self.fiid].view(-1, 1) == topk_items
        target = batch[self.frating].view(-1, 1)
        metric_dict = {}
        for cutoff in cutoffs:
            metric_dict[f'recall@{cutoff}'] = recall(pred, target, cutoff)
            metric_dict[f'ndcg@{cutoff}'] = ndcg(pred, target, cutoff)
        return metric_dict, bs
    
    def encode_target(self, target):
        return self.item_encoder(target)
    
    @property
    def item_vector(self):
        return self.item_encoder.weight[1:]