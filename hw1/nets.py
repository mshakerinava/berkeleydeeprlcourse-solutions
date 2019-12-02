import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlNet(nn.Module):
    def __init__(self, widths, act_fn, dropout_prob):
        super(ControlNet, self).__init__()
        assert len(widths) >= 2, '`widths` should at least contain input size and output size.'
        self.widths = widths
        self.act_fn = act_fn
        self.dropout_prob = dropout_prob
        self.depth = len(widths) - 1
        self.fc = nn.ModuleList()
        for i in range(self.depth - 1):
            self.fc.append(nn.Linear(in_features=widths[i], out_features=widths[i + 1]))
        self.fc_mu = nn.Linear(in_features=widths[-2], out_features=widths[-1])
        self.fc_log_var = nn.Linear(in_features=widths[-2], out_features=widths[-1])
        self.register_buffer('obs_mean', torch.zeros(self.widths[0], dtype=torch.float32))
        self.register_buffer('obs_std', torch.ones(self.widths[0], dtype=torch.float32))

    def get_device(self):
        # NOTE: This method only makes sense when all module parameters reside on the **same** device.
        return list(self.parameters())[0].device

    def set_obs_stats(self, obs_mean, obs_std):
        self.obs_mean = torch.tensor(obs_mean, device=self.obs_mean.device, dtype=torch.float32)
        self.obs_std = torch.tensor(obs_std + 0.1, device=self.obs_std.device, dtype=torch.float32)

    def forward(self, x):
        x = (x - self.obs_mean) / self.obs_std
        B = x.shape[0]
        assert list(x.shape) == [B, self.widths[0]]
        for i in range(self.depth - 1):
            x = self.fc[i](x)
            # NOTE: Order of applying `dropout` and `act_fn` can matter when `act_fn(0) != 0`.
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
            if 'training' in self.act_fn.__code__.co_varnames:
                x = self.act_fn(x, training=self.training)
            else:
                x = self.act_fn(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var

    def loss(self, mu, log_var, y):
        B = y.shape[0]
        assert list(mu.shape) == [B, self.widths[-1]]
        assert list(log_var.shape) == [B, self.widths[-1]]
        assert list(y.shape) == [B, self.widths[-1]]
        nll = 0.5 * log_var + (y - mu) ** 2 / (2 * torch.exp(log_var) + 1e-5)
        assert list(nll.shape) == [B, self.widths[-1]]
        return torch.mean(nll)
