import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ASDLoss(nn.Module):
    def __init__(self):
        super(ASDLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, out, target, mu=None, logvar=None):
        mse_loss = self.mse_loss(out, target)
        if (mu is not None) and (logvar is not None):
            kl_loss = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu**2)
            loss = mse_loss + kl_loss
        else:
            loss = mse_loss
        return loss