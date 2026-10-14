"""
Authors: Craig Fouts
License: Apache 2.0 license
"""

import torch

def transport_loss(x, y, dim=128, scale=1.):
    z = torch.normal(0., 1., (x.shape[1], dim))
    z /= z.square().sum(0, keepdim=True).sqrt()
    y_, y = (x@z).T.sort()[0], (y@z).T.sort()[0]
    mask = (torch.linspace(0, y.shape[1] - 1, x.shape[0]) + .5).long()
    loss = scale*(y_ - y[:, mask]).square().sum()/dim

    return loss
