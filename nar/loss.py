import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as func

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.eps = 1e-07

    def forward(self, outputs, targets, T=None):
        if T is None: return func.cross_entropy(outputs, targets)

        targets_onehot = torch.zeros_like(outputs)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets.unsqueeze(-1), 1)

        outputs = self.softmax(outputs)
        outputs = torch.clamp(outputs, self.eps, 1.0 - self.eps)

        targets_plus = torch.mm(targets_onehot, T.float())
        loss = torch.sum(targets_plus * torch.log(outputs), dim=1)
        return -torch.mean(loss)
