import torch
from ..data import SeqDataset
from src.basemodel import BaseModel
from src.metric import ndcg, recall


class SASRec(BaseModel):
    
    def add_model_specific_args(parent_parser):
        parent_parser = BaseModel.add_model_specific_args(parent_parser)
        parent_parser.add_argument_group('SASRec')
        parent_parser.add_argument("--embed_dim", type=int, default=64, help='embededding dimension')
        parent_parser.add_argument("--hidden_size", type=int, default=128)
        parent_parser.add_argument("--layer_num", type=int, default=2)
        parent_parser.add_argument("--head_num", type=int, default=2)
        parent_parser.add_argument("--dropout_rate", type=float, default=0.5, help='dropout rate')
        return parent_parser
    
    def __init__(self, config, train_data) -> None:
        super().__init__(config, train_data)

        self.embed_dim = self.config['embed_dim']
        self.n_layers = self.config['layer_num']
        self.n_head = self.config['head_num']
        self.hidden_size = self.config['hidden_size']
        self.dropout_rate = self.config['dropout_rate']
        self.activation = self.config['activation'] # relu, gelu
        self.layer_norm_eps = self.config['layer_norm_eps']
        self.max_seq_len = train_data.config['max_seq_len']
        self.fiid = train_data.fiid
        self.frating = train_data.frating

        self.position_emb = torch.nn.Embedding(self.max_seq_len, self.embed_dim)
        transformer_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.n_head,
            dim_feedforward=self.hidden_size,
            dropout=self.dropout_rate,
            activation=self.activation,
            layer_norm_eps=self.layer_norm_eps,
            batch_first=True,
            norm_first=False
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=self.n_layers,
        )
        self.layer_norm = torch.nn.LayerNorm(self.embed_dim, eps=self.layer_norm_eps)
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.item_encoder = self._get_item_encoder(train_data)

    def get_dataset_class():
        return SeqDataset
    
    def _get_item_encoder(self, train_data):
        return torch.nn.Embedding(train_data.num_items, self.embed_dim, padding_idx=0)

    
    def construct_query(self, batch):
        user_hist = batch['in_' + self.fiid]
        seq_len = batch['seqlen']
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device)
        positions = positions.unsqueeze(0).expand_as(user_hist)
        position_embs = self.position_emb(positions)

        seq_embs = self.item_encoder(user_hist)
        input_embs = position_embs + seq_embs
        input_embs = self.layer_norm(input_embs)    # BxLxD
        input_embs = self.dropout(input_embs)

        mask4padding = user_hist==0 # BxL
        L = user_hist.size(-1)
        attention_mask = ~torch.tril(torch.ones((L, L), dtype=torch.bool, device=user_hist.device))
        attn_output = self.transformer_encoder(src=input_embs, mask=attention_mask, src_key_padding_mask=mask4padding)  # BxLxD

        gather_index = (seq_len-1).view(-1, 1, 1).expand(-1, -1, attn_output.shape[-1]) # Bx1xD
        query_output = attn_output.gather(dim=1, index=gather_index).squeeze(1) # BxD
        return query_output
    
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