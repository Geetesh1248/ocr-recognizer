import torch
from torch.nn import functional as F


def detection_loss(logits, target, valid):
    """Balanced BCE (hard negative mining) plus masked Dice overlap loss."""
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce_parts = []
    for loss, truth, mask in zip(losses, target, valid):
        positive = loss[(truth > .5) & (mask > .5)]
        negative = loss[(truth <= .5) & (mask > .5)]
        count = min(negative.numel(), max(256, positive.numel()*3))
        hard_negative = torch.topk(negative, count).values if count else negative
        denominator = max(1, positive.numel()+count)
        bce_parts.append((positive.sum()+hard_negative.sum())/denominator)
    probability = logits.sigmoid()*valid
    truth = target*valid
    intersection = (probability*truth).sum(dim=(1, 2, 3))
    dice = 1-(2*intersection+1)/(probability.sum(dim=(1, 2, 3))+truth.sum(dim=(1, 2, 3))+1)
    return torch.stack(bce_parts).mean()+dice.mean()
