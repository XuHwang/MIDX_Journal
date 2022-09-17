import torch
from ..data import LanguageModelDataset
from src.basemodel import BaseModel
from src.loss_func import SampledSoftmax
from src.metric import perplexity

class LSTM(BaseModel):

    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group('Transformer')
        parent_parser.add_argument("--embed_dim", type=int, default=200, help='embededding dimension')
        parent_parser.add_argument("--dropout", type=float, default=0.1, help='dropout rate')
        parent_parser.add_argument("--n_layer", type=int, default=2, help='number of LSTM layers')
        return parent_parser

    def __init__(self, config, train_data) -> None:
        super().__init__(config, train_data)
        self.embedding = torch.nn.Embedding(self.num_items, self.config['embed_dim'], padding_idx=0)
        self.lstm = torch.nn.LSTM(
            input_size = self.config['embed_dim'],
            hidden_size = self.config['embed_dim'],
            num_layers = self.config['n_layer'],
            dropout = self.config['dropout'],
            batch_first = True
        )
        self.out_embedding = torch.nn.Embedding(self.num_items, self.config['embed_dim'], padding_idx=0)

        self.loss_func = SampledSoftmax(temperature=self.config['temperature'])

    def get_dataset_class():
        return LanguageModelDataset

    def construct_query(self, batch):
        B, L = batch['input'].shape
        seq_emb = self.embedding(batch['input'])    # [B,L,D]
        lstm_out = self.lstm(seq_emb) # [B,L,D]
        return lstm_out[0]

    def encode_target(self, target):
        return self.out_embedding(target)

    def _test_step(self, batch):
        label = batch['target']
        bs = label.size(0)
        query = self.construct_query(batch)
        scores = self.score_fn(query, self.out_embedding.weight[1:])
        return {'log_ppl': perplexity(scores, label)}, bs

    @property
    def item_vector(self):
        return self.out_embedding.weight[1:]
