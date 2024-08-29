import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional
from .base import Sampler
from ..scorer import InnerProductScorer, EuclideanScorer, CosineScorer

def kmeans(X, K_or_center, max_iter=300, verbose=False):
    N = X.size(0)
    if isinstance(K_or_center, int):
        K = K_or_center
        C = X[torch.randperm(N)[:K]]
    else:
        K = K_or_center.size(0)
        C = K_or_center
    prev_loss = np.inf
    for iter in range(max_iter):
        dist = torch.sum(X * X, dim=-1, keepdim=True) - 2 * \
            (X @ C.T) + torch.sum(C * C, dim=-1).unsqueeze(0)
        assign = dist.argmin(-1)
        assign_m = X.new_zeros(N, K)
        assign_m[(range(N), assign)] = 1
        loss = torch.sum(torch.square(X - C[assign, :])).item()
        if verbose:
            print(f'step:{iter:<3d}, loss:{loss:.3f}')
        if (prev_loss - loss) < prev_loss * 1e-6:
            break
        prev_loss = loss
        cluster_count = assign_m.sum(0)
        C = (assign_m.T @ X) / cluster_count.unsqueeze(-1)
        empty_idx = cluster_count < .5
        ndead = empty_idx.sum().item()
        C[empty_idx] = X[torch.randperm(N)[:ndead]]
    return C, assign, assign_m, C[assign, :]


def construct_index(cd01, K):
    cd01, indices = torch.sort(cd01)
    cluster, count = torch.unique_consecutive(cd01, return_counts=True)
    count_all = torch.zeros(K + 1, dtype=torch.long, device=cd01.device)
    count_all[cluster + 1] = count
    indptr = count_all.cumsum(dim=-1)
    return indices, indptr


class MIDXProductSampler(Sampler):
    """
    Uniform sampling for the final items
    """

    def __init__(self, num_items, num_clusters, scorer_fn=None):
        assert scorer_fn is None or isinstance(scorer_fn, InnerProductScorer)
        super(MIDXProductSampler, self).__init__(num_items, scorer_fn)
        self.K = num_clusters
        self.residual_quantizer = False

    def update(self, item_embs, max_iter=100):
        if isinstance(self.scorer, CosineScorer):
            item_embs = F.normalize(item_embs, dim=-1)
        embs1, embs2 = torch.chunk(item_embs, 2, dim=-1)
        self.c0, cd0, cd0m, _ = kmeans(
            embs1, self.c0 if hasattr(self, 'c0') else self.K, max_iter)
        self.c1, cd1, cd1m, _ = kmeans(
            embs2, self.c1 if hasattr(self, 'c1') else self.K, max_iter)
        
        # for retreival probability, considering padding
        self.c0_, self.cd0 = self._update_paddings(self.c0, cd0)
        self.c1_, self.cd1 = self._update_paddings(self.c1, cd1)
        
        cd01 = cd0 * self.K + cd1
        self.indices, self.indptr = construct_index(cd01, self.K**2)
        self._update(item_embs, cd0m, cd1m)

    
    def _update_paddings(self, c, cd):
        c_ = torch.cat([c.new_zeros(1, c.size(1)), c], dim=0)
        cd_ = torch.cat([-cd.new_ones(1), cd], dim=0) + 1
        return c_, cd_        

    def _update(self, item_embs, cd0m, cd1m):
        if not isinstance(self.scorer, EuclideanScorer):
            self.wkk = cd0m.T @ cd1m
        else:
            norm = torch.exp(-0.5*torch.sum(item_embs**2, dim=-1))
            self.wkk = cd0m.T @ (cd1m * norm.view(-1, 1))
            # this is similar, to avoid log 0 !!! in case of zero padding
            self.p = torch.cat([norm.new_ones(1), norm], dim=0)
            self.cp = norm[self.indices]
            for c in range(self.K**2):
                start, end = self.indptr[c], self.indptr[c+1]
                if end > start:
                    cumsum = self.cp[start:end].cumsum(0)
                    self.cp[start:end] = cumsum / cumsum[-1]

    @torch.no_grad()
    def forward(self, query, num_neg, pos_items=None):
        # need_reshape = False
        # if query.dim() > 2:
        #     need_reshape = True
        #     shape = query.shape[:-1]
        #     query = query.view(-1, query.size(-1))
        #     pos_items = pos_items.view(-1)
        with torch.no_grad():
            if isinstance(self.scorer, CosineScorer):
                query = F.normalize(query, dim=-1)
            if self.residual_quantizer is True:
                q0, q1 = query.reshape(-1, query.size(-1)), query.reshape(-1, query.size(-1))
            else:
                q0, q1 = query.reshape(-1, query.size(-1)).chunk(2, dim=-1)
            r1 = q1 @ self.c1.T
            r1s = torch.softmax(r1, dim=-1)  # num_q x K1
            r0 = q0 @ self.c0.T
            r0s = torch.softmax(r0, dim=-1)  # num_q x K0
            s0 = (r1s @ self.wkk.T) * r0s  # num_q x K0 | wkk: K0 x K1
            k0 = torch.multinomial(
                s0, num_neg, replacement=True)  # num_q x neg
            p0 = torch.gather(r0, -1, k0)     # num_q * neg
            subwkk = self.wkk[k0, :]          # num_q x neg x K1
            s1 = subwkk * r1s.unsqueeze(1)     # num_q x neg x K1
            k1 = torch.multinomial(
                s1.view(-1, s1.size(-1)), 1).squeeze(-1).view(*s1.shape[:-1])  # num_q x neg
            p1 = torch.gather(r1, -1, k1)  # num_q x neg
            k01 = k0 * self.K + k1  # num_q x neg
            p01 = p0 + p1
            neg_items, neg_prob = self.sample_item(k01, p01)
            if pos_items is not None:
                pos_prob = None if pos_items is None else self.compute_item_p(
                    query, pos_items)
                return pos_prob, neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)
            else:
                return neg_items.view(*query.shape[:-1], -1), neg_prob.view(*query.shape[:-1], -1)

    def sample_item(self, k01, p01, pos=None):
        # num_q x neg, the number of items
        item_cnt = self.indptr[k01 + 1] - self.indptr[k01]
        item_idx = torch.floor(
            item_cnt * torch.rand_like(item_cnt.float())).int()  # num_q x neg
        neg_items = self.indices[item_idx + self.indptr[k01]] + 1
        neg_prob = p01
        return neg_items, neg_prob



    def _sample_item_with_pop(self, k01, p01):
        # k01 num_q x neg, p01 num_q x neg
        start = self.indptr[k01]
        last = self.indptr[k01 + 1] - 1
        count = last - start + 1
        maxlen = count.max()
        fullrange = start.unsqueeze(-1) + torch.arange(
            maxlen, device=start.device).reshape(1, 1, maxlen)  # num_q x neg x maxlen
        fullrange = torch.minimum(fullrange, last.unsqueeze(-1))
        # @todo replace searchsorted with torch.bucketize
        item_idx = torch.searchsorted(self.cp[fullrange], torch.rand_like(
            p01).unsqueeze(-1)).squeeze(-1)  # num_q x neg
        # item_idx = torch.minimum(item_idx, last)
        neg_items = self.indices[item_idx + self.indptr[k01]] + 1
        # plus 1 due to considering padding, since p include num_items + 1 entries
        neg_probs = self.p[neg_items]
        return neg_items, p01 + torch.log(neg_probs)
        
        
    def _sample_item_with_pop(self, k01, p01):
        # k01 num_q x neg, p01 num_q x neg
        start = self.indptr[k01]
        last = self.indptr[k01 + 1] - 1
        count = last - start + 1
        maxlen = count.max()
        fullrange = start.unsqueeze(-1) + torch.arange(
            maxlen, device=start.device).reshape(1, 1, maxlen)  # num_q x neg x maxlen
        fullrange = torch.minimum(fullrange, last.unsqueeze(-1))
        # @todo replace searchsorted with torch.bucketize
        item_idx = torch.searchsorted(self.cp[fullrange], torch.rand_like(
            p01).unsqueeze(-1)).squeeze(-1)  # num_q x neg
        # item_idx = torch.minimum(item_idx, last)
        neg_items = self.indices[item_idx + self.indptr[k01]] + 1
        # plus 1 due to considering padding, since p include num_items + 1 entries
        neg_probs = self.p[neg_items]
        return neg_items, p01 + torch.log(neg_probs)
        

    def compute_item_p(self, query, pos_items):
        # query: B x L x D, pos_items: B x L || query: B x D, pos_item: B x L1 || assume padding=0
        if pos_items.dim() == 1:
            pos_items_ = pos_items.unsqueeze(1)
        else:
            pos_items_ = pos_items
        k0 = self.cd0[pos_items_]  # B x L || B x L1
        k1 = self.cd1[pos_items_]  # B x L || B x L1
        c0 = self.c0_[k0, :]  # B x L x D || B x L1 x D
        c1 = self.c1_[k1, :]  # B x L x D || B x L1 x D
        if self.residual_quantizer is True:
            q0, q1 = query, query
        else:
            q0, q1 = query.chunk(2, dim=-1)  # B x L x D || B x D
        if query.dim() == pos_items_.dim():
            r = (torch.bmm(c0, q0.unsqueeze(-1)) +
                 torch.bmm(c1, q1.unsqueeze(-1))).squeeze(-1)  # B x L1
        else:
            r = (q0 * c0).sum(-1) + (q1 * c1).sum(-1)
        if not hasattr(self, 'p'):
            return r.view_as(pos_items)
        else:
            return (r + torch.log(self.p[pos_items_])).view_as(pos_items)

class MIDXResidualSampler(MIDXProductSampler):
    def __init__(self, num_items, num_clusters, scorer_fn=None):
        super(MIDXResidualSampler, self).__init__(num_items, num_clusters, scorer_fn)
        self.residual_quantizer = True

    def update(self, item_embs, max_iter=100):
        if isinstance(self.scorer, CosineScorer):
            item_embs = F.normalize(item_embs, dim=-1)
        
        self.c0, cd0, cd0m, item_embs_quant = kmeans(
            item_embs, self.c0 if hasattr(self, 'c0') else self.K, max_iter)
        self.c1, cd1, cd1m, _ = kmeans(
            item_embs - item_embs_quant, self.c1 if hasattr(self, 'c1') else self.K, max_iter)
        
        # for retreival probability, considering padding
        self.c0_, self.cd0 = self._update_paddings(self.c0, cd0)
        self.c1_, self.cd1 = self._update_paddings(self.c1, cd1)
        
        cd01 = cd0 * self.K + cd1
        self.indices, self.indptr = construct_index(cd01, self.K**2)
        self._update(item_embs, cd0m, cd1m)



class MIDXSamplerLearnProduct(MIDXProductSampler):
    def __init__(self, num_items, num_clusters, emb_dim, scorer_fn=None):
        super().__init__(num_items, num_clusters, scorer_fn)

        self.residual_quantizer = False
        assert emb_dim % 2 == 0, ValueError("embedding dimension should be even")
        self.c0 = nn.Parameter(torch.FloatTensor(num_clusters, emb_dim//2))
        nn.init.normal_(self.c0, std=0.01)
        self.c1 = nn.Parameter(torch.FloatTensor(num_clusters, emb_dim//2))
        nn.init.normal_(self.c1, std=0.01)

    
    def encoding(self, emb_vector:torch.Tensor, hard:bool=False):
        """
        encode the emb_vector with the vq-codebook
        
        here we assume we have two codebooks
        """
        emb1, emb2 = torch.chunk(emb_vector, 2, dim=-1)
        res1 = self._encode(emb1, self.c0, hard)
        res2 = self._encode(emb2, self.c1, hard)
        return torch.cat([res1, res2], dim=-1)


    # def _encode(self, emb_vec, codebook, hard=False):
    #     # Version_V1: use the l2 distance
    #     # emb_vec: B x L x D
    #     # codebook: K x D
    #     dist_subspace = torch.cdist(emb_vec, codebook) # B x L x K
    #     logit = torch.exp(-dist_subspace/2)
    #     weight_hard = torch.nn.functional.gumbel_softmax(logit, tau=1.0, hard=True) # B x L x K
    #     weight_soft = torch.nn.functional.gumbel_softmax(logit, tau=1.0, hard=False)
    #     if hard:
    #         weight = weight_hard.detach()
    #     else:
    #         weight = weight_soft.detach()
    #     output = (weight.unsqueeze(-1) * codebook).sum(-2)
    #     return output
    
    def _encode(self, emb_vec, codebook, hard=False):
        # V2: use the inner product to calculate the similarity
        # emb_vec: B x L x D or N x D
        # codebook: K x D
        if emb_vec.dim() == codebook.dim():
            # N x D,  K x D
            logit = torch.matmul(emb_vec, codebook.T) # N x K
        else:
            if emb_vec.dim() == 3:
                # B x L x D,  K x D
                logit = torch.einsum('bld,kd->blk', emb_vec, codebook)
            elif emb_vec.dim() == 4:
                # B x L x N x D,  K x D
                logit = torch.einsum('blnd,kd->blnk', emb_vec, codebook)
        if hard:
            weight = torch.nn.functional.gumbel_softmax(logit, tau=1.0, hard=True, dim=-1).detach()
        else:
            weight = torch.nn.functional.gumbel_softmax(logit, tau=1.0, hard=False, dim=-1).detach()
        return (weight.unsqueeze(-1) * codebook).sum(-2)
        

    def index_update(self, X, center, **kwargs):
        N = X.size(0)
        dist = torch.matmul(X, center.T)
        if dist.isinf().any():
            print("Warning: inf distance found!!!")
            import pdb; pdb.set_trace()
        assign = dist.argmax(-1)
        assign_m = X.new_zeros(N, self.K)
        assign_m[(range(N), assign)] = 1
        
        reconstruct_X = center[assign, :]
        return center, assign, assign_m, reconstruct_X

    def update(self, item_embs, max_iter=100):
        if isinstance(self.scorer, CosineScorer):
            item_embs = F.normalize(item_embs, dim=-1)
        embs1, embs2 = torch.chunk(item_embs, 2, dim=-1)
        self.c0, cd0, cd0m, _ = self.index_update(embs1, self.c0)
        self.c1, cd1, cd1m, _ = self.index_update(embs2, self.c1)

        # for retreival probability, considering padding
        self.c0_, self.cd0 = self._update_paddings(self.c0, cd0)
        self.c1_, self.cd1 = self._update_paddings(self.c1, cd1)

        cd01 = cd0 * self.K + cd1
        self.indices, self.indptr = construct_index(cd01, self.K**2)
        self._update(item_embs, cd0m, cd1m)

class MIDXSamplerLearnResidual(MIDXSamplerLearnProduct):
    def __init__(self, num_items, num_clusters, emb_dim, scorer_fn=None):
        super(MIDXSamplerLearnProduct, self).__init__(num_items, num_clusters, scorer_fn)

        self.residual_quantizer = True
        self.c0 = nn.Parameter(torch.FloatTensor(num_clusters, emb_dim))
        nn.init.normal_(self.c0, std=0.01)
        self.c1 = nn.Parameter(torch.FloatTensor(num_clusters, emb_dim))
        nn.init.normal_(self.c1, std=0.01)

    
    def encoding(self, emb_vector:torch.Tensor, hard:bool=False):
        """
        encode the emb_vector with the vq-codebook
        
        here we assume we have two codebooks

        emb_vector: B x L x D
        """
        res1 = self._encode(emb_vector, self.c0, hard) # K x D or B x L x D
        res2 = self._encode(emb_vector - res1, self.c1, hard)
        return  torch.add(res1, res2)


    def update(self, item_embs, max_iter=100):
        if isinstance(self.scorer, CosineScorer):
            item_embs = F.normalize(item_embs, dim=-1)
        self.c0, cd0, cd0m, reconstruct_X1 = self.index_update(item_embs, self.c0)
        self.c1, cd1, cd1m, _ = self.index_update(item_embs - reconstruct_X1, self.c1)

        # for retreival probability, considering padding
        self.c0_, self.cd0 = self._update_paddings(self.c0, cd0)
        self.c1_, self.cd1 = self._update_paddings(self.c1, cd1)

        cd01 = cd0 * self.K + cd1
        self.indices, self.indptr = construct_index(cd01, self.K**2)
        self._update(item_embs, cd0m, cd1m)