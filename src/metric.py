import torch
import torch.nn.functional as F 

def perplexity(pred, target):
    pred = pred.view(-1, pred.size(-1))
    target = target.view(-1)
    ppl = F.nll_loss(torch.log_softmax(pred, dim=-1), target, ignore_index=0, reduction='mean')
    return ppl
    