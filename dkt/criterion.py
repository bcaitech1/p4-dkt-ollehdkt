
import torch.nn as nn


def get_criterion(pred, target):
    loss = nn.BCELoss(reduction="none")
    # loss = nn.CrossEntropyLoss(reduction="none")
    return loss(pred, target)