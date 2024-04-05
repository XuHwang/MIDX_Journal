import torch
import torch.nn.functional as F

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


class RotatELoss(torch.nn.Module):
    def __init__(self, margin=10.0, weight=False, sqrt=False) -> None:
        super().__init__()
        self.margin = margin
        self.weight = weight
        self.sqrt = sqrt

    def forward(self, pos_score, log_pos_prob, neg_score, log_neg_prob):
        pos_dist, neg_dist = - pos_score, - neg_score
        if self.sqrt:
            pos_dist = torch.sqrt(- pos_dist)
            neg_dist = torch.sqrt(- neg_dist)
        pos_dist = F.logsigmoid(self.margin - pos_dist)
        if self.weight:
            # diff = pos_score.view(-1, 1) - neg_score
            weight = torch.softmax(log_neg_prob, dim=-1).detach()
            # weight = torch.softmax(neg_score - log_neg_prob, dim=-1).detach()
            neg_dist = F.logsigmoid(neg_dist - self.margin) * weight
        else:
            K = neg_score.size(-1)
            neg_dist = F.logsigmoid(neg_dist - self.margin) / K
        neg_dist_sum = torch.sum(neg_score, dim=-1)
        loss = (- pos_dist - neg_dist_sum) / 2
        return loss.mean()

    def extra_repr(self) -> str:
        return f"margin={self.margin}, weight={self.weight}"


class TransELoss(RotatELoss):
    def __init__(self, margin=1.0, weight=False, sqrt=False) -> None:
        super().__init__(margin, weight, sqrt)
    
    def forward(self, pos_score, log_pos_prob, neg_score, log_neg_prob):
        # score is negative distance. For distance, the smaller the better.
        pos_dist, neg_dist = - pos_score, - neg_score
        if self.sqrt:
            pos_dist = torch.sqrt(pos_dist)
            neg_dist = torch.sqrt(neg_dist)
        diff = self.margin + pos_dist.view(-1, 1) - neg_dist    # [B, N]
        diff[diff<0] = 0.0
        if self.weight:
            weight = torch.softmax(neg_score - log_neg_prob, dim=-1).detach()
        else:
            weight = 1 / neg_score.size(-1)
        loss = (weight * diff).sum(-1)
        return loss.mean()


class PRISLoss(torch.nn.Module):
    def __init__(self, margin=0.0, weight=False, sqrt=False) -> None:
        super().__init__()
        self.margin = margin
        self.weight = weight
        self.sqrt = sqrt
    
    def forward(self, pos_score, log_pos_prob, neg_score, log_neg_prob):
        # score is negative distance. For distance, the smaller the better.
        pos_dist, neg_dist = - pos_score, - neg_score
        if self.sqrt:
            pos_dist = torch.sqrt(pos_dist)
            neg_dist = torch.sqrt(neg_dist)
        diff = - pos_dist.view(-1, 1) + neg_dist    # [B, N]
        if self.weight:
            weight = torch.softmax(neg_score - log_neg_prob, dim=-1).detach()
        else:
            weight = 1 / neg_score.size(-1)
        loss = - (weight * F.logsigmoid(diff-self.margin)).sum(-1)
        return loss.mean()