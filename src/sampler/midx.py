import torch
import numpy as np
import torch.nn.functional as F
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
    return C, assign, assign_m, loss


def construct_index(cd01, K):
    cd01, indices = torch.sort(cd01)
    cluster, count = torch.unique_consecutive(cd01, return_counts=True)
    count_all = torch.zeros(K + 1, dtype=torch.long, device=cd01.device)
    count_all[cluster + 1] = count
    indptr = count_all.cumsum(dim=-1)
    return indices, indptr


class MIDXSamplerUniform(Sampler):
    """
    Uniform sampling for the final items
    """

    def __init__(self, num_items, num_clusters, scorer_fn=None):
        assert scorer_fn is None or isinstance(scorer_fn, InnerProductScorer)
        super(MIDXSamplerUniform, self).__init__(num_items, scorer_fn)
        self.K = num_clusters

    def update(self, item_embs, max_iter=100):
        if isinstance(self.scorer, CosineScorer):
            item_embs = F.normalize(item_embs, dim=-1)
        embs1, embs2 = torch.chunk(item_embs, 2, dim=-1)
        self.c0, cd0, cd0m, _ = kmeans(
            embs1, self.c0 if hasattr(self, 'c0') else self.K, max_iter)
        self.c1, cd1, cd1m, _ = kmeans(
            embs2, self.c1 if hasattr(self, 'c1') else self.K, max_iter)
        # for retreival probability, considering padding
        self.c0_ = torch.cat(
            [self.c0.new_zeros(1, self.c0.size(1)), self.c0], dim=0)
        # for retreival probability, considering padding
        self.c1_ = torch.cat(
            [self.c1.new_zeros(1, self.c1.size(1)), self.c1], dim=0)
        # for retreival probability, considering padding
        self.cd0 = torch.cat([-cd0.new_ones(1), cd0], dim=0) + 1
        # for retreival probability, considering padding
        self.cd1 = torch.cat([-cd1.new_ones(1), cd1], dim=0) + 1
        cd01 = cd0 * self.K + cd1
        self.indices, self.indptr = construct_index(cd01, self.K**2)
        self._update(item_embs, cd0m, cd1m)

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
        # TODO: remove positive items
        if not hasattr(self, 'cp'):
            # num_q x neg, the number of items
            item_cnt = self.indptr[k01 + 1] - self.indptr[k01]
            item_idx = torch.floor(
                item_cnt * torch.rand_like(item_cnt.float())).int()  # num_q x neg
            neg_items = self.indices[item_idx + self.indptr[k01]] + 1
            neg_prob = p01
            return neg_items, neg_prob
        else:
            return self._sample_item_with_pop(k01, p01)
            # return self._sample_item_with_pop_large(k01, p01) # for those datasets with extremely large number of items

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

    def _sample_item_with_pop_large(self, k01, p01):
        # the earlier version may exceed the cuda memory when the number of candidate corpus grows extremely large
        # the reason lies in the huge tensor fullrange, with the shape of num_q x neg x maxlen, when maxlen is huge [unbalanced clusters]
        # k01 num_q x neg, p01 num_q x neg
        union_c, inverse_indices, counts = k01.view(-1).unique(return_counts=True, return_inverse=True) 
        neg_items = torch.zeros_like(k01.view(-1))
        neg_probs = torch.zeros_like(p01.view(-1))
        
        start = self.indptr[union_c] # K^2
        last = self.indptr[union_c + 1] - 1 # K^2 
        maxlen = (last - start + 1).max() 
        fullrange = start.unsqueeze(-1)  + torch.arange(maxlen, device=k01.device).reshape(1, maxlen)
        fullrange = torch.minimum(fullrange, last.unsqueeze(-1)) # K^2 x maxlen
        item_idx = torch.searchsorted(self.cp[fullrange], torch.rand(size=(union_c.shape[0], counts.max()), device=p01.device)) # K^2 x max_count
        item_idx = torch.minimum(item_idx, (last - start).unsqueeze(-1)) 
        items = self.indices[item_idx + self.indptr[union_c].unsqueeze(-1)] + 1
        probs = self.p[items]

        for idx in range(union_c.shape[0]):
            mask = torch.eq(inverse_indices, idx)
            neg_items[mask] = items[idx][:mask.sum()]
            neg_probs[mask] = probs[idx][:mask.sum()]

        return neg_items.view(*k01.shape), p01 + torch.log(neg_probs.view(*p01.shape))
        

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
        q0, q1 = query.chunk(2, dim=-1)  # B x L x D || B x D
        if query.dim() == pos_items_.dim():
            r = (torch.bmm(c0, q0.unsqueeze(-1)) +
                 torch.bmm(c1, q1.unsqueeze(-1))).squeeze(-1)  # B x L1
        else:
            r = (q0 * c0).sum(-1) + (q1 * c1).sum(-1)
            # pos_items_ = pos_items_.unsqueeze(1)
        if not hasattr(self, 'p'):
            return r.view_as(pos_items)
        else:
            return (r + torch.log(self.p[pos_items_])).view_as(pos_items)


class MIDXSamplerPop(MIDXSamplerUniform):
    """
    Popularity sampling for the final items
    """

    def __init__(self, pop_count: torch.Tensor, num_clusters, scorer=None, mode=1):
        super(MIDXSamplerPop, self).__init__(
            pop_count.shape[0], num_clusters, scorer)
        if mode == 0:
            pop_count = torch.log(pop_count + 1)
        elif mode == 1:
            pop_count = torch.log(pop_count + 1) + 1e-6
        elif mode == 2:
            pop_count = pop_count**0.75
        self.pop_count = torch.nn.Parameter(pop_count[:-1], requires_grad=False) # TODO: check 

    def _update(self, item_embs, cd0m, cd1m):
        if not isinstance(self.scorer, EuclideanScorer):
            norm = self.pop_count
        else:
            norm = self.pop_count * \
                torch.exp(-0.5*torch.sum(item_embs**2, dim=-1))
        self.wkk = cd0m.T @ (cd1m * norm.view(-1, 1))
        # self.p = torch.from_numpy(np.insert(pop_count, 0, 1.0))
        # this is similar, to avoid log 0 !!! in case of zero padding
        self.p = torch.cat([norm.new_ones(1), norm], dim=0)
        self.cp = norm[self.indices]
        for c in range(self.K**2):
            start, end = self.indptr[c], self.indptr[c+1]
            if end > start:
                cumsum = self.cp[start:end].cumsum(0)
                self.cp[start:end] = cumsum / cumsum[-1]