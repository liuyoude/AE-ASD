import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
import torchaudio


# linear block
class Liner_Module(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Liner_Module, self).__init__()
        self.liner = nn.Linear(input_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        x = self.liner(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


# AE
class Auto_encoder(nn.Module):
    def __init__(self, input_dim=640, output_dim=640):
        super(Auto_encoder, self).__init__()
        self.encoder = nn.Sequential(
            Liner_Module(input_dim=input_dim, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=8),
        )
        self.decoder = nn.Sequential(
            Liner_Module(input_dim=8, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            nn.Linear(128, output_dim),
        )

    def forward(self, input: torch.Tensor):
        x_feature = self.encoder(input)
        x = self.decoder(x_feature)
        return x, x_feature


# VAE
class VAE(nn.Module):
    def __init__(self, input_dim=640, output_dim=640):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            Liner_Module(input_dim=input_dim, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
        )
        self.fc_mean = Liner_Module(input_dim=128, out_dim=8)
        self.fc_logvar = Liner_Module(input_dim=128, out_dim=8)
        self.fc_z = Liner_Module(input_dim=128, out_dim=8)

        self.decoder = nn.Sequential(
            Liner_Module(input_dim=8, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            Liner_Module(input_dim=128, out_dim=128),
            nn.Linear(128, output_dim),
        )

    # def reparameterization(self, mu, logvar):
    #     std = torch.exp(0.5 * logvar)
    #     rand = torch.randn(std.size()).to(std.device)
    #     z = rand * std + mu
    #     return z

    def reparameterization(self, z, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # rand = torch.randn(std.size()).to(std.device)
        return z * std + mu

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mean(h)
        logvar = self.fc_logvar(h)
        z = self.fc_z(h)
        z = self.reparameterization(z, mu, logvar)
        output = self.decoder(z)
        if self.training:
            return output, z, mu, logvar
        else:
            return output, z

