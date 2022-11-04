import torch, sys
import torch.nn.functional as F 

def perplexity(pred, target):
    pred = pred.view(-1, pred.size(-1))
    target = target.view(-1)
    ppl = F.nll_loss(torch.log_softmax(pred, dim=-1), target, ignore_index=0, reduction='mean')
    return ppl

def mr(diff):
    r""" Mean Rank
    Args:
        diff: [B, num_items], scores of all items minus pos score
    """
    rank = (diff > 0).float().sum(-1) + 1
    return rank.mean()


def mrr(diff):
    r""" Mean Reciprocal Rank
    Args:
        diff: [B, num_items], scores of all items minus pos score
    """
    rank = (diff > 0).float().sum(-1) + 1
    return (1.0 / rank).mean()


def hits(diff, k):
    r"""Calculate the Hits.
    Args:
        diff: [B, num_items], scores of all items minus pos score
    """
    rank = (diff > 0).float().sum(-1) + 1
    hits = (rank <= k).float().mean()
    return hits

def precision(pred, target, k):
    r"""Calculate the precision.
    Precision are defined as:
    .. math::
        Precision = \frac{TP}{TP+FP}
    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values. 
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.
        target(torch.FloatTensor): [B, num_target]. The ground truth.
    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    output = pred[:, :k].sum(dim=-1).float() / k
    return output.mean()

def recall(pred, target, k):
    r"""Calculating recall.
    Recall value is defined as below:
    .. math::
        Recall= \frac{TP}{TP+FN} 
    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values. 
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.
        target(torch.FloatTensor): [B, num_target]. The ground truth.
    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    count = (target > 0).sum(-1)
    output = pred[:, :k].sum(dim=-1).float() / count
    return output.mean()


def _dcg(pred, k):
    k = min(k, pred.size(1))
    denom = torch.log2(torch.arange(k).type_as(pred) + 2.0).view(1, -1)
    return (pred[:, :k] / denom).sum(dim=-1)


def ndcg(pred, target, k):
    r"""Calculate the Normalized Discounted Cumulative Gain(NDCG).
    Args:
        pred(torch.BoolTensor): [B, num_items]. The prediction result of the model with bool type values. 
            If the value in the j-th column is `True`, the j-th highest item predicted by model is right.
        target(torch.FloatTensor): [B, num_target]. The ground truth.
    Returns:
        torch.FloatTensor: a 0-dimensional tensor.
    """
    pred_dcg = _dcg(pred.float(), k)
    ideal_dcg = _dcg(torch.sort((target > 0).float(), descending=True)[0], k) 
    all_irrelevant = torch.all(target <= sys.float_info.epsilon, dim=-1)
    pred_dcg[all_irrelevant] = 0
    pred_dcg[~all_irrelevant] /= ideal_dcg[~all_irrelevant]
    return pred_dcg.mean()
