"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

import torch.nn as nn
import torch_geometric.nn as gnn

NORMALIZATION = {'batch': nn.BatchNorm1d, 'layer': nn.LayerNorm}
ACTIVATION = {'relu': nn.ReLU, 'prelu': nn.PReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'softplus': nn.Softplus}

class MLP(nn.Sequential):
    """Implementation of a multilayer perceptron.
    
    Parameters
    ----------
    channels : int
        Number of channels in each layer.
    bias : bool, default=True
        Whether to include additive bias.
    normalization : str, default='batch'
        Normalization function.
    activation : str, default='relu'
        Activation function.
    dropout : float, default=0.0
        Amount of dropout.

    Attributes
    ----------
    None

    Usage
    -----
    >>> channels = (in_channels, hidden_channels, ..., out_channels)
    >>> model = MLP(*channels, **kwargs)
    >>> output = model(data)
    """

    def __init__(self, *channels, bias=True, normalization='batch', activation='relu', dropout=0.):
        modules = []

        for i in range(1, len(channels) - 1):
            modules.append(self.layer(channels[i - 1], channels[i], bias, normalization, activation, dropout))

        modules.append(nn.Linear(channels[-2%len(channels)], channels[-1], bias))

        super().__init__(*modules)

    @staticmethod
    def layer(in_channels, out_channels, bias=True, normalization='batch', activation='relu', dropout=0.):
        """Constructs a single neural network layer with optional normalization,
        activation, and dropout modules.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        bias : bool, default=True
            Whether to include additive bias.
        normalization : str, default='batch'
            Normalization function.
        activation : str, default='relu'
            Activation function.
        dropout : float, default=0.0
            Amount of dropout.

        Returns
        -------
        Sequential
            Neural network layer module.
        """
        
        modules = nn.Sequential(nn.Linear(in_channels, out_channels, bias))

        if normalization is not None:
            modules.append(NORMALIZATION[normalization](out_channels))

        if activation is not None:
            modules.append(ACTIVATION[activation]())

        if dropout > 0.:
            modules.append(nn.Dropout(dropout))

        return modules
    
class GCN(gnn.MessagePassing):
    """Implementation of a graph convolutional network. Loosly based on methods
    proposed by Guohao Li, Chenxin Xiong, Ali Thabet, and Bernard Ghanem.

    https://doi.org/10.48550/arXiv.2006.07739

    Parameters
    ----------
    in_channels : int
        Number of channels in each input layer.
    out_channels : int | tuple | list, default=None
        Number of channels in each output layer.
    bias : bool, default=False
        Whether to include additive bias.
    normalization : str, default=None
        Normalization function.
    activation : str, default=None
        Activation function.
    dropout : float, default=0.0
        Amount of dropout.
    aggregation : str, default='softmax'
        Message aggregation function.
    offset : float
        Message offset.
    kwargs : dict
        Additional MessagePassing arguments.

    Attributes
    ----------
    root_model : MLP
        Pre-propagation neural network.
    bias_model : MLP
        Propagation bias neural network.
    out_model : MLP
        Post-propagation neural network.

    Usage
    -----
    >>> channels = (in_channels, hidden_channels, ..., out_channels)
    >>> model = GCN(*channels, **kwargs)
    >>> output = model(data, edges)
    """
    
    def __init__(self, *in_channels, out_channels=None, bias=False, normalization='batch', activation='relu', dropout=0., aggregation='softmax', offset=1e-7, **kwargs):
        super().__init__(aggr=aggregation, **kwargs)

        self.out_channels = (out_channels,) if isinstance(out_channels, int) else out_channels
        self.offset = offset

        self.root_model = MLP(*in_channels, bias=bias, normalization=normalization, activation=activation, dropout=dropout)
        self.bias_model = MLP(*in_channels, bias=bias, normalization=normalization, activation=activation, dropout=dropout)

        if out_channels is not None:
            self.out_model = MLP(in_channels[-1], *self.out_channels, bias=bias, normalization=normalization, activation=activation, dropout=dropout)

    def forward(self, x, edges):
        """Performs a single forward pass through the network.
        
        Parameters
        ----------
        x : Tensor
            Node features.
        edges : Tensor
            Edge indices.

        Returns
        -------
        Tensor
            Network output.
        """
        
        root, bias = self.root_model(x), self.bias_model(x)
        out = self.propagate(edges, x=(root, x)) + bias

        if hasattr(self, 'out_model'):
            out = self.out_model(out)

        return out

    def message(self, x_j):
        """Constructs the message emitted by the given node.
        
        Parameters
        ----------
        x_j : Tensor
            Node features.

        Returns
        -------
        Tensor
            Node message.
        """
        
        message = x_j.relu() + self.offset

        return message
