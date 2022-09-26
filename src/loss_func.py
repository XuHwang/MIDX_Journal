import torch

class FullSoftmax(torch.nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.T = temperature

    def forward(self, pos_score, full_score):
        keep_dim = False if full_score.dim() > pos_score.dim() else True
        # full_score: B x N, pos_score: B [KG task]
        # full_score: B x L x N, pos_score: B x L [LM, paddings]
        # full_score: B x N, pos_score: B x D [EC, paddings]
        output = torch.logsumexp(full_score, dim=-1, keepdim=keep_dim) - pos_score
        notpadnum = torch.logical_not(torch.isinf(pos_score)).float().sum(-1)
        output = torch.nan_to_num(output, posinf=0).sum(-1) / notpadnum
        return torch.mean(output)


class SampledSoftmax(torch.nn.Module):
    def __init__(self, temperature=1):
        super().__init__()
        self.T = temperature

    def forward(self, pos_score, log_pos_prob, neg_score, log_neg_prob):
        new_pos = pos_score - log_pos_prob
        new_neg = neg_score - log_neg_prob
        if new_pos.dim() < new_neg.dim():
            new_pos.unsqueeze_(-1)
        new_neg = torch.cat([new_pos, new_neg], dim=-1)
        output = torch.logsumexp(new_neg, dim=-1, keepdim=True) - new_pos
        if new_pos.dim() > 2:
            # For LM models:
            new_pos.squeeze_(-1)
            output.squeeze_(-1)
        notpadnum = torch.logical_not(torch.isinf(new_pos)).float().sum(-1)
        output = torch.nan_to_num(output, posinf=0, nan=0).sum(-1) / notpadnum
        return torch.mean(output)