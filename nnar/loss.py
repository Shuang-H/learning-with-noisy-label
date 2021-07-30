import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as func

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.eps = 1e-07

    def forward(self, outputs, targets, weights=None):
        if weights is None: return func.cross_entropy(outputs, targets)

        outputs = self.softmax(outputs)
        outputs = torch.clamp(outputs, self.eps, 1.0 - self.eps)
        loss = torch.sum(weights * torch.log(outputs), dim=1)
        return -torch.mean(loss)
