import torch
from ..data import LanguageModelDataset
from src.basemodel import BaseModel
from src.loss_func import SampledSoftmax
from src.metric import perplexity

class Transformer(BaseModel):

    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group('Transformer')
        parent_parser.add_argument("--d_model", type=int, default=64, help='embededding dimension')
        parent_parser.add_argument("--n_head", type=int, default=4, help='number of head in attention')
        parent_parser.add_argument("--dim_feedforward", type=int, default=128, help='dimension of feedforward')
        parent_parser.add_argument("--dropout", type=float, default=0.1, help='dropout rate')
        parent_parser.add_argument("--activation", type=str, default='gelu', help='activation')
        parent_parser.add_argument("--n_layer", type=int, default=4, help='number of transformer layers')
        return parent_parser

    def __init__(self, config, train_data) -> None:
        super().__init__(config, train_data)
        self.embedding = torch.nn.Embedding(self.num_items, self.config['embed_dim'], padding_idx=0)
        self.position_embedding = torch.nn.Embedding(train_data.max_seq_len+1, self.config['embed_dim'], padding_idx=0)
        tfm_layer = torch.nn.TransformerEncoderLayer(
            d_model = self.config['embed_dim'],
            nhead = self.config['n_head'],
            dim_feedforward = self.config['dim_feedforward'],
            dropout = self.config['dropout'],
            activation = self.config['activation'],
            batch_first = True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(tfm_layer, self.config['n_layer'])
        self.loss_func = SampledSoftmax(temperature=self.config['temperature'])

    def get_dataset_class():
        return LanguageModelDataset

    def construct_query(self, batch):
        B, L = batch['input'].shape
        src_padding_mask = batch['input'] == 0
        seq_emb = self.embedding(batch['input'])    # [B,L,D]
        position = torch.arange(1, L+1, device=seq_emb.device)
        position = position.view(1, -1).expand((B, -1))   # [B,L]
        position = position.masked_fill(src_padding_mask, 0)
        pos_emb = self.position_embedding(position) #[B,L,D]
        tfm_input = seq_emb + pos_emb
        src_mask = torch.triu(torch.ones((L, L),device=seq_emb.device),diagonal=1).bool()
        tfm_out = self.transformer_encoder(tfm_input, mask=src_mask, src_key_padding_mask=src_padding_mask) # [B,L,D]
        return tfm_out

    def encode_target(self, target):
        return self.embedding(target)

    def _test_step(self, batch):
        label = batch['target']
        bs = label.size(0)
        query = self.construct_query(batch)
        scores = self.score_fn(query, self.embedding.weight[1:])
        return {'log_ppl': perplexity(scores, label)}, bs

    @property
    def item_vector(self):
        return self.embedding.weight[1:]
