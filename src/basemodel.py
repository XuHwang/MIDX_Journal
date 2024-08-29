import logging, os
import itertools
import torch
from torch import optim
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch.nn.functional as F

from .scorer import InnerProductScorer,EuclideanScorer
from .loss_func import FullSoftmax, SampledSoftmax
from .utils import color_dict
from .sampler import (UniformSampler, PopularSampler,
                      SphereSampler, RFFSampler,
                      SphereSamplerAppr, RffSamplerAppr,
                      LSHSampler,
                      MIDXProductSampler, MIDXResidualSampler, MIDXSamplerLearnResidual, MIDXSamplerLearnProduct)

from .init import normal_initialization

class BaseModel(LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group('MIDX')
        parent_parser.add_argument("--learning_rate", type=float, default=0.001, help='learning rate')
        parent_parser.add_argument("--learner", type=str, default="adam", help='optimization algorithm')
        parent_parser.add_argument("--scheduler", type=str, default="none", help='lr scheduler algorithm')
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
        parent_parser.add_argument('--num_cluster', type=int, default=16, help='number of codewords for midx-based samplers')
        parent_parser.add_argument('--num_neg', type=int, default=50, help='the number of negative samples')
        parent_parser.add_argument('--pop_mode', type=int, default=1, help='the mode for pop')
        parent_parser.add_argument('--sphere_alpha', type=float, default=100, help='alpha for sphere kernel sampler')
        parent_parser.add_argument('--rff_temp', type=float, default=4.0, help='temp for rff sampler')
        parent_parser.add_argument('--rff_dim', type=int, default=32, help='temp for rff sampler')
        parent_parser.add_argument('--lsh_bits', type=int, default=4, help='number of bits for lsh sampler')
        parent_parser.add_argument('--lsh_tables', type=int, default=16, help='number of tables for lsh sampler')
        parent_parser.add_argument('--sampler_update_step', type=int, default=100, help='update frequency for sampler')
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

        if hasattr(train_data, "num_feat"):
            self.console_logger.info(f"Number of Feat: {train_data.num_feat}")
            self.num_feat = train_data.num_feat # work for extreme classification task
        
        if hasattr(train_data, "num_users"):    # Recommendation task
            self.console_logger.info(f"Number of Users: {train_data.num_users}")
            self.console_logger.info(f"Number of Inters: {train_data.num_inters}")
            self.console_logger.info(f"Sparsity: {1-train_data.num_inters/(train_data.num_items*train_data.num_users)}")

    @staticmethod
    def get_dataset_class():
        pass

    def construct_query(self, batch):
        pass

    def encode_target(self, target):
        pass

    def on_fit_start(self) -> None:
        super().on_fit_start()
        if not hasattr(self, '_init_param'):
            for name, module in self.named_children():
                init_method = normal_initialization(self.config['init_range'])
                module.apply(init_method)

    def sampling(self, query, num_neg, pos_item):
        # query: [B,D], pos_item: [B, D] 
        # query: [B,L,D], pos_item: [B,L,D]
        if self.sampler is not None:
            return self.sampler(query, num_neg, pos_item)
        else:
            raise NotImplementedError("To be implemented.")

    def configure_optimizers(self):
        params = self.parameters()
        if isinstance(self.sampler, MIDXSamplerLearnProduct) or isinstance(self.sampler, MIDXSamplerLearnResidual):
            params_sampler = self.sampler.parameters()
            params = itertools.chain(params, params_sampler)
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
        save_dir = os.path.dirname(self.console_logger.handlers[1].baseFilename)
        save_dir = os.path.join(save_dir, 'ckpt' )
        ckp_callback = ModelCheckpoint(dirpath=save_dir, save_top_k=1, mode=self.config['early_stop_mode'], save_last=True)
        return [ckp_callback, early_stopping]

    def configure_sampler(self):
        if self.config['sampler'] == 'midx-pq':
            return MIDXProductSampler(self.num_items, self.config['num_cluster'], self.score_fn)
        elif self.config['sampler'] == 'uni':
            return UniformSampler(self.num_items, self.score_fn)
        elif self.config['sampler'] == 'pop':
            return PopularSampler(self.item_freq, self.score_fn, mode=self.config['pop_mode'])
        elif self.config['sampler'] == 'sphere':
            return SphereSampler(self.num_items, self.score_fn, alpha=self.config['sphere_alpha'])
        elif self.config['sampler'] == 'rff':
            return RFFSampler(self.num_items, self.score_fn, temp=self.config['rff_temp'], rff_dim=self.config['rff_dim'])
        elif self.config['sampler'] == 'sphere_a':
            return SphereSamplerAppr(self.num_items, self.score_fn, alpha=self.config['sphere_alpha'])
        elif self.config['sampler'] == 'rff_a':
            return RffSamplerAppr(self.num_items, self.score_fn)
        elif self.config['sampler'] == 'lsh':
            return LSHSampler(
                num_items=self.num_items,
                n_dims=self.config['embed_dim'],
                n_bits=self.config['lsh_bits'],
                n_table=self.config['lsh_tables'],
                device=self.device,
                scorer_fn=self.score_fn
                )
        elif (self.config['sampler'] is None) or (self.config['sampler']=='none'):
            return None
        elif self.config['sampler'] =='midx-rq':
            return MIDXResidualSampler(self.num_items, self.config['num_cluster'],self.score_fn)
        elif self.config['sampler'] == 'midx-learn-pq':
            return MIDXSamplerLearnProduct(self.num_items, self.config['num_cluster'], self.config['embed_dim'], self.score_fn)
        elif self.config['sampler'] == 'midx-learn-rq':
            return MIDXSamplerLearnResidual(self.num_items, self.config['num_cluster'], self.config['embed_dim'],  self.score_fn)
        else:
            raise ValueError(f"Not supported for such sampler {self.config['sampler']}.")

    def forward(self, batch, pad2inf=True):
        output = {}
        output_quantizer = {}
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
            output['log_pos_prob'] = log_pos_prob.detach()
            output['log_neg_prob'] = log_neg_prob.detach()
        else: # full softmax
            output['full_score'] = self.score_fn(query, self.item_vector)
        
        sampler_class_list = [MIDXSamplerLearnProduct, MIDXSamplerLearnResidual]
        if any([isinstance(self.sampler, cls) for cls in sampler_class_list]):
            # get the quantization loss for optimization
            # pos_vec_ = self.sampler.encoding(pos_vec.detach())
            # neg_vec_ = self.sampler.encoding(neg_vec.detach())

            # Trial 1: || x_i - x'_i ||^2, it works, similar to the uni
            # dis1 = torch.norm(pos_vec.detach() - pos_vec_, dim=-1).mean()
            # dis2 = torch.norm(neg_vec.detach() - neg_vec_, dim=-1).mean()
            # output_quantizer['reconstract_loss'] = dis1 + dis2

            # Trial 2: ||r_ui - r'_ui||^2,  smaller coefficent for the loss
            # pos_score_ = self.score_fn(query.detach(), pos_vec_)
            # neg_score_ = self.score_fn(query.detach(), neg_vec_)
            # if pad2inf:
            #     pos_score_[batch['target']==0] = -float('inf')
            # dis1 = torch.nn.functional.mse_loss(pos_score_, output['pos_score'].detach())
            # dis2 = torch.nn.functional.mse_loss(neg_score_, output['neg_score'].detach())
            # output_quantizer['reconstract_loss'] = dis1 + dis2

            # Trial 3:
            # pos_score_ = self.score_fn(query.detach(), pos_vec_)
            # if pad2inf:
            #     pos_score_[batch['target']==0] = -float('inf')
            # output_quantizer['pos_score'] = pos_score_
            # output_quantizer['neg_score'] = self.score_fn(query.detach(), neg_vec_)
            # output_quantizer['log_pos_prob'] = log_pos_prob.detach()
            # output_quantizer['log_neg_prob'] = log_neg_prob.detach()
            # output_quantizer['reconstract_loss'] = self.loss_fn(**output_quantizer)

            # Trial 5:
            # Calculate the KL-distance between the original softmax probability and the quantized softmax probability
            item_vec_ = self.sampler.encoding(self.item_vector.detach())

            full_score = self.score_fn(query.detach(), self.item_vector.detach())
            full_sp = F.log_softmax(full_score, dim=-1).detach() # avoid inf or nan
            full_score_quant = self.score_fn(query.detach(), item_vec_)
            full_quant_sp = F.log_softmax(full_score_quant, dim=-1) # avoid inf or nan
            N = self.item_vector.size(0)
            loss_kl = torch.nn.functional.kl_div(full_sp.reshape(-1, N), full_quant_sp.reshape(-1, N), reduction='batchmean', log_target=True) # target is log_softmax
            dis = torch.nn.PairwiseDistance()(self.item_vector.detach(), item_vec_).mean()
            output_quantizer['reconstract_loss'] = dis
            output_quantizer['kl_div'] = loss_kl
        return output, output_quantizer

    def configure_loss(self):
        if self.sampler is not None:
            return SampledSoftmax()
        else:
            return FullSoftmax()

    def on_train_start(self) -> None:
        if self.sampler is not None:
            self.sampler.update(self.item_vector)
    

    def training_step(self, batch, batch_idx):
        if self.sampler is not None:
            self.sampler.update(self.item_vector)
        output, output_q = self.forward(batch)
        if output_q:
            loss = self.loss_fn(**output)
            loss_q = output_q['reconstract_loss']
            loss_kl = output_q['kl_div']
            self.log('ssl_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('recons_loss', loss_q, on_step=False, on_epoch=True, prog_bar=True)
            self.log('kl_loss', loss_kl, on_step=False, on_epoch=True, prog_bar=True)
            return {"loss": loss + loss_q + loss_kl} 
        else:
            loss = self.loss_fn(**output)
            self.log('ssl_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        # if (self.current_epoch % 10 == 0) and (batch_idx == 0):
        #     query = self.construct_query(batch)
        #     save_path = './save_items/vec/query_vector_epoch{}.pt'.format(self.current_epoch) 
        #     torch.save(query.detach().cpu(), save_path)
        return self._test_step(batch)

    def test_step(self, batch, batch_idx):
        return self._test_step(batch)

    def _test_step(self, batch):
        # need to override to calculate metrics
        pass

    def on_train_epoch_start(self) -> None:
        if self.sampler is not None:
            self.sampler.update(self.item_vector)
        # if (self.trainer.current_epoch % 10) == 0:
        #     save_path = './save_items/vec/item_vector_epoch{}.pt'.format(self.trainer.current_epoch) 
        #     torch.save(self.item_vector.cpu(), save_path)

    def training_epoch_end(self, outputs):   
        loss_metric = {'train_'+ k: torch.hstack([e[k] for e in outputs]).mean() for k in outputs[0]}
        self.log_dict(loss_metric)
        output_dict = self.trainer.logged_metrics
        output_dict.update({'epoch': self.trainer.current_epoch})
        self.console_logger.info(color_dict(output_dict, False))
        if self.config['mode'] == 'tune':
            metric = {}
            for k, v in output_dict.items():
                if isinstance(v, torch.Tensor):
                    metric[k] = v.item()
                else:
                    metric[k] = v
            metric['default'] = metric[self.config['monitor_metric']]
            # nni.report_intermediate_result(metric)

    def validation_epoch_end(self, outputs):
        metric_dict = self._eval_epoch_end(outputs)
        self.log_dict(metric_dict)
        return metric_dict

    def test_epoch_end(self, outputs):
        metric_dict = self._eval_epoch_end(outputs)
        self.log_dict(metric_dict)
        self.console_logger.info(color_dict(self.trainer.logged_metrics, False))
        if self.config['mode'] == 'tune':
            metric = {}
            for k, v in metric_dict.items():
                if isinstance(v, torch.Tensor):
                    metric[k] = v.item()
                else:
                    metric[k] = v
            metric['default'] = metric[self.config['monitor_metric']]
            # nni.report_final_result(metric)
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
            elif self.config['scheduler'].lower() == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=10,
                    eta_min=0.0
                )
            else:
                scheduler = None
        else:
            scheduler = None
        return scheduler

    @property
    def item_vector(self):
        return None