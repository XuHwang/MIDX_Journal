import torch
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
    