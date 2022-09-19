import logging
from typing import Union, Dict, Tuple, List
from torch import Tensor

import torch
from torch import optim
import pytorch_lightning
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from .scorer import InnerProductScorer
from .loss_func import FullSoftmax, SampledSoftmax
from .utils import SAVE_DIR, color_dict
from .sampler import MIDXSamplerUniform, MIDXSamplerPop

class BaseModel(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group('MIDX')
        parent_parser.add_argument("--learning_rate", type=float, default=0.001, help='learning rate')
        parent_parser.add_argument("--learner", type=str, default="adam", help='optimization algorithm')
        parent_parser.add_argument('--weight_decay', type=float, default=0, help='weight decay coefficient')
        parent_parser.add_argument('--epochs', type=int, default=50, help='training epochs')
        parent_parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parent_parser.add_argument('--eval_batch_size', type=int, default=128, help='evaluation batch size')
        parent_parser.add_argument('--val_n_epoch', type=int, default=1, help='valid epoch interval')
        parent_parser.add_argument('--early_stop_patience', type=int, default=10, help='early stop patience')
        parent_parser.add_argument('--gpu', type=int, action='append', default=None, help='gpu number')
        parent_parser.add_argument('--init_method', type=str, default='xavier_normal', help='init method for model')
        parent_parser.add_argument('--init_range', type=float, help='init range for some methods like normal')
        parent_parser.add_argument('--sampler', type=str, default=None, help='which sampler to use')
        return parent_parser


    def __init__(self, config, train_data) -> None:
        super().__init__()
        self.config = config
        self.target_vector = None
        if self.config['monitor_metric'] is None:
            self.val_metric = 'train_loss'
        else:
            self.val_metric = self.config['monitor_metric']
        self.num_items = train_data.num_items
        self.item_freq = train_data.item_freq
        self.score_fn = InnerProductScorer()

        self.sampler = self.configure_sampler()
        self.loss_fn = self.configure_loss()
        
        self.console_logger = logging.getLogger('MIDX')
        self.console_logger.info(f"Number of Items: {self.num_items}")
        
    @staticmethod
    def get_dataset_class():
        pass

    def construct_query(self, batch):
        pass

    def encode_target(self, target):
        pass

    def sampling(self, query, num_neg, pos_item):
        # query: [B,D], pos_item: [B, D] 
        # query: [B,L,D], pos_item: [B,L,D]
        if self.sampler is not None:
            return self.sampler(query, num_neg, pos_item)
        else:
            raise NotImplementedError("To be implemented.")

    def configure_optimizers(self):
        params = self.parameters()
        optimizer = self.get_optimizer(params)
        scheduler = self.get_scheduler(optimizer)
        m = self.val_metric
        if scheduler:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': m,
                    'interval': 'epoch',
                    'frequency': 1,
                    'strict': False
                }
            }
        else:
            return optimizer

    def configure_callbacks(self):
        monitor_metric = self.config['monitor_metric']
        # self.val_metric = next(iter(eval_metric)) if isinstance(eval_metric, list)  else eval_metric
        # cutoffs = self.config['cutoff'] if isinstance(self.config['cutoff'], list) else [self.config['cutoff']]
        # if len(eval.get_rank_metrics(self.val_metric)) > 0:
        #     self.val_metric += '@' + str(cutoffs[0])
        early_stopping = EarlyStopping(monitor_metric, verbose=True, patience=self.config['early_stop_patience'], mode=self.config['early_stop_mode'])
        import os
        ckpt_name = os.path.basename(self.console_logger.handlers[1].baseFilename).split('.')[0]
        save_dir = os.path.join(SAVE_DIR, ckpt_name)
        ckp_callback = ModelCheckpoint(dirpath=save_dir, save_top_k=1, mode=self.config['early_stop_mode'], save_last=True)
        return [ckp_callback, early_stopping]

    def configure_sampler(self):
        if self.config['sampler'] == 'midx-uni':
            return MIDXSamplerUniform(self.num_items, 8, self.score_fn)
        elif self.config['sampler'] == 'midx-pop':
            return MIDXSamplerPop(self.item_freq, 8, self.score_fn)
        elif self.config['sampler'] is None:
            return None
        else:
            raise ValueError(f"Not supported for such sampler {self.config['sampler']}.")

    def forward(self, batch, pad2inf=True):
        output = {}
        query = self.construct_query(batch)
        pos_item = batch['target']
        pos_vec = self.encode_target(pos_item)
        pos_score = self.score_fn(query, pos_vec)
        if pad2inf:
            pos_score[batch['target']==0] = -float('inf')
        output['pos_score'] = pos_score

        if self.sampler is not None:    # sampled softmax
            log_pos_prob, neg_id, log_neg_prob = self.sampling(query, self.config['num_neg'], pos_item)
            neg_vec = self.encode_target(neg_id)
            output['neg_score'] = self.score_fn(query, neg_vec)
            output['log_pos_prob'] = log_pos_prob
            output['log_neg_prob'] = log_neg_prob
        else: # full softmax
            output['full_score'] = self.score_fn(query, self.item_vector)
        return output

    def configure_loss(self):
        if self.sampler is not None:
            return SampledSoftmax()
        else:
            return FullSoftmax()
        
    def on_train_start(self) -> None:
        if self.sampler is not None:
            self.sampler.update(self.item_vector)

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        loss = self.loss_fn(**output)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        return self._test_step(batch)

    def test_step(self, batch, batch_idx):
        return self._test_step(batch)

    def _test_step(self, batch):
        # need to override to calculate metrics
        pass

    def training_epoch_end(self, outputs):   
        loss_metric = {'train_'+ k: torch.hstack([e[k] for e in outputs]).mean() for k in outputs[0]}
        self.log_dict(loss_metric)
        output_dict = self.trainer.logged_metrics
        output_dict.update({'epoch': self.trainer.current_epoch})
        self.console_logger.info(color_dict(output_dict, False))

    def validation_epoch_end(self, outputs):
        metric_dict = self._eval_epoch_end(outputs)
        self.log_dict(metric_dict)
        return metric_dict

    def test_epoch_end(self, outputs):
        metric_dict = self._eval_epoch_end(outputs)
        self.log_dict(metric_dict)
        self.console_logger.info(color_dict(self.trainer.logged_metrics, False))
        return metric_dict

    def _eval_epoch_end(self, outputs):
        metric_dict = {k: 0.0 for k in outputs[0][0].keys()}
        total_bs = 0
        for metric, bs in outputs:
            for k,v in metric.items():
                metric_dict[k] += v * bs
            total_bs += bs
        metric_dict = {k: v/total_bs for k,v in metric_dict.items()}
        if 'log_ppl' in metric_dict:
            metric_dict['ppl'] = torch.exp(metric_dict['log_ppl'])
        return metric_dict

    def get_optimizer(self, params):
        r"""Return optimizer for specific parameters.
        The optimizer can be configured in the config file with the key ``learner``. 
        Supported optimizer: ``Adam``, ``SGD``, ``AdaGrad``, ``RMSprop``, ``SparseAdam``.
        .. note::
            If no learner is assigned in the configuration file, then ``Adam`` will be user.
        Args:
            params: the parameters to be optimized.
        
        Returns:
            torch.optim.optimizer: optimizer according to the config.
        """
        '''@nni.variable(nni.choice(0.1, 0.05, 0.01, 0.005, 0.001), name=learning_rate)'''
        learning_rate = self.config['learning_rate']
        '''@nni.variable(nni.choice(0.1, 0.01, 0.001, 0), name=decay)'''
        decay = self.config['weight_decay']
        if self.config['learner'].lower() == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=decay)
        elif self.config['learner'].lower() == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=decay)
        elif self.config['learner'].lower() == 'adagrad':
            optimizer = optim.Adagrad(params, lr=learning_rate, weight_decay=decay)
        elif self.config['learner'].lower() == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=learning_rate, weight_decay=decay)
        elif self.config['learner'].lower() == 'sparse_adam':
            optimizer = optim.SparseAdam(params, lr=learning_rate)
            #if self.weight_decay > 0:
            #    self.logger.warning('Sparse Adam cannot argument received argument [{weight_decay}]')
        else:
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer

    def get_scheduler(self, optimizer):
        r"""Return learning rate scheduler for the optimizer.
        Args:
            optimizer(torch.optim.Optimizer): the optimizer which need a scheduler.
        Returns:
            torch.optim.lr_scheduler: the learning rate scheduler.
        """
        if self.config['scheduler'] is not None:
            if self.config['scheduler'].lower() == 'exponential':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
            elif self.config['scheduler'].lower() == 'onplateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            else:
                scheduler = None
        else:
            scheduler = None
        return scheduler

    @property
    def item_vector(self):
        return None