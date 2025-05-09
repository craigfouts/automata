"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

import torch.nn as nn
import torch_geometric.nn as gnn

NORMALIZATION = {'batch': nn.BatchNorm1d, 'layer': nn.LayerNorm}
ACTIVATION = {'relu': nn.ReLU, 'prelu': nn.PReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'softplus': nn.Softplus}

class MLP(nn.Sequential):
    """Implementation of a multilayer perceptron (MLP) model.
    
    Parameters
    ----------
    channels : int
        Number of channels in each layer.
    bias : bool, default=True
        Whether to include a hidden additive bias term.
    final_bias : bool, default=True
        Whether to include a final additive bias term.
    normalization : str, default='batch'
        Hidden normalization function.
    final_norm : str, default=None
        Final normalization function.
    activation : str, default='relu'
        Hidden activation function.
    final_act : str, default=None
        Final activation function.
    dropout : float, default=0.0
        Amount of hidden dropout.
    final_drop : float, default=0.0
        Amount of final dropout.

    Attributes
    ----------
    None

    Usage
    -----
    >>> channels = (input_dim, hidden_dim, ..., output_dim)
    >>> model = MLP(*channels, **kwargs)
    >>> output = model(input)
    """

    def __init__(self, *channels, bias=True, final_bias=True, normalization='batch', final_norm=None, activation='relu', final_act=None, dropout=0., final_drop=0.):
        modules = []

        for i in range(1, len(channels) - 1):
            modules.extend(self.layer(channels[i - 1], channels[i], bias, normalization, activation, dropout))

        modules.extend(self.layer(channels[-2], channels[-1], final_bias, final_norm, final_act, final_drop))

        super().__init__(*modules)

    @staticmethod
    def layer(in_channels, out_channels, bias=True, normalization='batch', activation='relu', dropout=0.):
        """Constructs a neural network layer with optional normalization, 
        activation, and dropout modules.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        bias : bool, default=True
            Whether to include an additive bias term.
        normalization : str, default='batch'
            Normalization function.
        activation : str, default='relu'
            Activation function.
        dropout : float, default=0.0
            Amount of dropout.

        Returns
        -------
        list
            Layer modules.
        """

        modules = [nn.Linear(in_channels, out_channels, bias)]

        if normalization is not None:
            modules.append(NORMALIZATION[normalization](out_channels))

        if activation is not None:
            modules.append(ACTIVATION[activation]())
        
        if dropout > 0.:
            modules.append(nn.Dropout(dropout))

        return modules

class GCN(gnn.MessagePassing):
    """Implementation of a graph convolutional network (GCN) model.
    
    Parameters
    ----------
    channels : int
        Number of channels in each layer.
    aggregation : str
        Message aggregation function.
    offset : float
        Message offset.
    kwargs : dict
        Network parameters.

    Attributes
    ----------
    model : MLP
        Network model.

    Usage
    -----
    >>> channels = (input_dim, hidden_dim, ..., output_dim)
    >>> model = GCN(*channels, **kwargs)
    >>> output = model(input, edges)
    """
    
    def __init__(self, *channels, aggregation='softmax', offset=1e-7, **kwargs):
        super().__init__(aggr=aggregation)

        self.offset = offset

        self.model = MLP(*channels, **kwargs)

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
        
        out = self.model(2*self.propagate(edges, x=(x, x)))

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
            Message.
        """
        
        message = x_j.relu() + self.offset

        return message
