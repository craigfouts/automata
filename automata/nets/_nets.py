"""
Authors: Craig Fouts
License: Apache 2.0 license
"""

from torch import nn, optim
from torch_geometric import nn as gnn
from ..utils import get_kwargs
from ..utils.sugar import attrmethod

ACTS = {'relu': nn.ReLU, 'prelu': nn.PReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'softplus': nn.Softplus, 'softmax': nn.Softmax}
NORMS = {'batch': nn.BatchNorm1d, 'layer': nn.LayerNorm}
OPTIMS = {'adam': optim.Adam, 'sgd': optim.SGD}

class MLP(nn.Sequential):
    @attrmethod
    def __init__(self, *channels, bias=True, norm=None, act=None, drop=0., final_bias=True, final_norm=None, final_act=None, final_drop=0., **kwargs):
        modules = []

        for i in range(1, len(channels) - 1):
            modules.append(self.layer(channels[i - 1], channels[i], bias, norm, act, drop, **kwargs))

        modules.append(self.layer(channels[-2%len(channels)], channels[-1], final_bias, final_norm, final_act, final_drop, **kwargs))
        super().__init__(*modules)

    @staticmethod
    def layer(in_channels, out_channels=None, bias=True, norm=None, act=None, drop=0., **kwargs):
        if out_channels is None:
            out_channels = in_channels

        layer_kwargs = dict(tuple(locals().items())[2:-1], **kwargs)
        modules = [nn.Linear(in_channels, out_channels, bias)]

        if norm is not None:
            norm_kwargs = get_kwargs(norm := NORMS[norm], **layer_kwargs)
            modules.append(norm(out_channels, **norm_kwargs))

        if act is not None:
            act_kwargs = get_kwargs(act := ACTS[act], **layer_kwargs)
            modules.append(act(**act_kwargs))

        if drop > 0.:
            modules.append(nn.Dropout(drop))

        module = nn.Sequential(*modules)

        return module

class GCN(gnn.MessagePassing):
    """Graph convolutional network module loosely based on methods proposed by 
    (Li et al., 2006) [https://doi.org/10.48550/arXiv.2006.07739].
    """

    @attrmethod
    def __init__(self, *channels, out_channels=None, bias=False, norm='batch', act='relu', drop=0., aggr='softmax', offset=1e-7, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        if isinstance(out_channels, int):
            self.out_channels = (out_channels,)

        self.root_net = MLP(*channels, bias=bias, norm=norm, act=act, drop=drop)
        self.bias_net = MLP(*channels, bias=bias, norm=norm, act=act, drop=drop)

        if out_channels is not None:
            self.out_net = MLP(channels[-1], *self.out_channels, bias=bias, norm=norm, act=act, drop=drop)

    def forward(self, x, edges):
        root, bias = self.root_net(x), self.bias_net(x)
        out = self.propagate(edges, x=(root, x)) + bias

        if hasattr(self, 'out_net'):
            out = self.out_net(out)

        return out

    def message(self, x_j):
        msg = x_j.relu() + self.offset

        return msg
