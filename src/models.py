"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

import torch
import torch.nn as nn
import warnings
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch_cluster import knn
from tqdm import tqdm
from nets import GCN
from util import show_progress, transport_loss

class NPA(nn.Module):
    def __init__(self, n_neighbors=8, mlp_channels=None, out_channels=None, loss=transport_loss, warn=True, **kwargs):
        super().__init__(**kwargs)

        self.n_neighbors = n_neighbors
        self.mlp_channels = (mlp_channels,) if isinstance(mlp_channels, int) else mlp_channels
        self.out_channels = (out_channels,) if isinstance(out_channels, int) else out_channels
        self.loss = loss
        self.warn = warn

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_log = []

    def build(self, in_channels , n_epochs=4000, learning_rate=1e-1):
        channels = (in_channels,)

        if self.mlp_channels is not None:
            channels = self.mlp_channels + channels

        if self.out_channels is not None:
            channels += self.out_channels + (in_channels,)

        self.model = GCN(in_channels, channels[0], out_channels=channels[1:])
        self.optimizer = Adam(self.parameters(), learning_rate)
        self.scheduler = OneCycleLR(self.optimizer, learning_rate, n_epochs)

        return self
    
    def step(self, x, target):
        loss = self.loss(x, target)

        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        elif self.warn:
            warnings.warn('Optimizer not initialized.')

        if self.scheduler is not None:
            self.scheduler.step()
        elif self.warn:
            warnings.warn('Scheduler not initialized.')

        return loss.item()
    
    def forward(self, x, target=None, update_rate=1e-3, n_steps=1):
        if self.model is not None:
            for _ in range(n_steps):
                edges = knn(x[:, :2], x[:, :2], self.n_neighbors + 1)
                update = self.model(x, edges)
                x = x + update*update_rate
        elif self.warn:
            warnings.warn('Model not initialized.')

        if target is not None:
            loss = self.step(x[:, :5], target)
            return x, loss

        return x
    
    def fit(self, seed, target, n_epochs=4000, learning_rate=1e-1, update_rate=1e-3, n_steps=64, step_rate=1e-2, verbosity=2, display_rate=1):
        self.build(seed.shape[-1], n_epochs, learning_rate)
        min_steps, max_steps = n_steps, n_steps + 1

        for i in tqdm(range(n_epochs)) if verbosity > 1 else range(n_epochs):
            x, loss = self(seed, target, update_rate, n_steps)
            self.loss_log.append(loss)

            if verbosity > 1 and i%display_rate == 0:
                show_progress(x.detach(), self.loss_log, f'steps: {n_steps}', f'epochs: {i + 1}')

            max_steps += step_rate
            n_steps = torch.randint(min_steps, int(max_steps), (1,)).item()

        return self

# class Model(nn.Module):

#     def __init__(self, loss_function=transport_loss, **kwargs):
#         super().__init__(**kwargs)

#         self.loss_function = loss_function

#         self.loss_log = []

#     def build(self):
#         return self

#     def forward(self, x, n_steps):
#         return x
    
#     def step(self, x, target):
#         loss = self.loss_function(x, target)

#         if hasattr(self, 'optimizer'):
#             loss.backward()
#             self.optimizer.step()
#             self.optimizer.zero_grad()

#         if hasattr(self, 'scheduler'):
#             self.scheduler.step()

#         return loss.item()
    
#     def display(self, x, epoch, n_steps, display_rate=10):
#         if display_rate%epoch == 0:
#             show_progress(x.detach(), self.loss_log, f'steps: {n_steps}', f'epochs: {epoch + 1}')

#     def fit(self, x, target, n_epochs=10000, learning_rate=1e-1, n_steps=64, step_rate=0., update_rate=1e-3, display_rate=10, verbosity=1, **kwargs):
#         args, min_steps, max_steps = locals(), n_steps, n_steps + 1
#         self.build(*[args[key] for key in self.build.__code__.co_varnames])
#         forward_args = [args[key] for key in self.forward.__code__.co_varnames if key != 'x']
#         step_args = [args[key] for key in self.step.__code__.co_varnames if key not in ('x', 'n_steps')]
#         display_args = [args[key] for key in self.display.__code__.co_varnames if key not in ('x', 'epoch')]
        
#         for i in tqdm(range(n_epochs)) if verbosity > 0 else range(n_epochs):
#             x = self(x, *forward_args)
#             self.loss_log.append(self.step(x, n_steps, *step_args))
#             self.display(x, i, n_steps, *display_args)
#             max_steps += step_rate
#             n_steps = torch.randint(min_steps, int(max_steps), (1,)).item()

#         return self
    
# class NPA(Model):
#     def __init__(self, n_neighbors=8, channels=32, **kwargs):
#         super().__init__(**kwargs)

#         self.n_neighbors = n_neighbors
#         self.channels = (channels,) if isinstance(channels, int) else channels
        
#         self.loss_log = []

#     def build(self, x, n_epochs, learning_rate):
#         in_channels = x.shape[-1]
#         self.model = gnn.Sequential('x, edges', [
#             (GCN(in_channels, self.channels[0]), 'x, edges -> x1'),
#             (MLP(*self.channels, in_channels), 'x1 -> x2')
#         ])
#         self.optimizer = Adam(self.parameters(), learning_rate)
#         self.scheduler = OneCycleLR(self.optimizer, learning_rate, n_epochs)

#         return self
    
#     def forward(self, x, n_steps=1, update_rate=1e-3):
#         for _ in range(n_steps):
#             edges = knn(x[:, :2], x[:, :2], self.n_neighbors + 1)
#             update = self.model(x, edges)
#             x = x + update*update_rate

#         return x

# class NPA(nn.Module):
#     def __init__(self, n_neighbors=8, channels=32, loss_function=transport_loss, **kwargs):
#         super().__init__(**kwargs)

#         self.n_neighbors = n_neighbors
#         self.channels = (channels,) if isinstance(channels, int) else channels
#         self.loss_function = loss_function
        
#         self.model = None
#         self.optimizer = None
#         self.scheduler = None
#         self.loss_log = []
#         self.max_steps = None

#     def build(self, seed, n_epochs=1000, learning_rate=1e-1):
#         in_channels = seed.shape[-1]
#         self.model = gnn.Sequential('x, edges', [
#             (GCN(in_channels, self.channels[0]), 'x, edges -> x1'),
#             (MLP(*self.channels, in_channels), 'x1 -> x2')
#         ])
#         self.optimizer = Adam(self.parameters(), learning_rate)
#         self.scheduler = OneCycleLR(self.optimizer, learning_rate, n_epochs)

#         return self
    
#     def step(self, x, target):
#         loss = self.loss_function(x, target)
#         loss.backward()
#         self.optimizer.step()
#         self.scheduler.step()
#         self.optimizer.zero_grad()

#         return loss.item()
    
#     def forward(self, x, target=None, update_rate=1e-3, n_steps=1):
#         for _ in range(n_steps):
#             edges = knn(x[:, :2], x[:, :2], self.n_neighbors + 1)
#             update = self.model(x, edges)
#             x = x + update*update_rate

#         if target is not None:
#             self.loss_log.append(self.step(x, target))

#         return x
    
#     def fit(self, seed, target, n_epochs=1000, learning_rate=1e-1, update_rate=1e-3, n_steps=64, step_rate=1e-1, verbosity=2, display_rate=1):
#         self.build(seed, n_epochs, learning_rate)
#         min_steps, self.max_steps = n_steps, n_steps + 1

#         for i in tqdm(range(n_epochs)) if verbosity > 0 else range(n_epochs):
#             x = self(x, target, update_rate, n_steps)

#             if verbosity > 1 and display_rate%i == 0:
#                 show_progress(x.detach(), self.loss_log, f'steps: {n_steps}', f'epochs: {i + 1}')

#             if i > n_epochs/2:
#                 self.max_steps += step_rate
#                 n_steps = torch.randint(min_steps, int(self.max_steps), (1,)).item()

#         return self

#     def transform(self, _=None):
#         parameters = self.parameters()

#         return parameters
    
#     def predict(self, x):
#         out = self(x, n_steps=self.max_steps if self.max_steps is not None else 1)

#         return out

# class NPA(nn.Module):
#     """Implementation of a basic neural particle automata (NPA) model.
    
#     Parameters
#     ----------
#     n_neighbors : int, default=9
#         Number of neighbors for each particle.
#     n_layers : int, default=3
#         Number of network layers.
#     hidden_scale : int, default=2
#         Scale of hidden network layers.
#     loss_function : func, default=transport_loss
#         Loss function.
#     kwargs : dict
#         Network parameters.

#     Attributes
#     ----------
#     model : GCN
#         Network model.
#     optimizer : Module
#         Network optimizer.
#     scheduler : Module
#         Learning rate scheduler.
#     loss_log : list
#         Record of the total loss for each step.

#     Usage
#     -----
#     >>> model = NPA(*args, **kwargs)
#     >>> model.train(seed, target, *args, **kwargs)
#     >>> out = model(seed, *args, **kwargs)
#     """
    
#     def __init__(self, n_neighbors=9, hidden_channels=(128, 128), loss_function=transport_loss, **kwargs):
#         super().__init__()

#         self.n_neighbors = n_neighbors
#         self.hidden_channels = (hidden_channels,) if isinstance(hidden_channels, int) else hidden_channels
#         self.loss_function = loss_function
#         self.kwargs = kwargs

#         self.model = None
#         self.optimizer = None
#         self.scheduler = None
#         self.loss_log = []

#     def build(self, seed, n_epochs=10000, learning_rate=1e-2):
#         """Initializes model parameters and class attributes.
        
#         Parameters
#         ----------
#         seed : Tensor
#             Initial state.
#         n_epochs : int, default=10000
#             Number of training epochs.
#         learning_rate : float, default=1e-2
#             Training step size.
        
#         Returns
#         -------
#         self
#             I return therefore I am.
#         """
        
#         in_channels = seed.shape[-1]
#         # hidden_channels = [in_channels*self.hidden_scale for _ in range(self.n_layers - 1)]
#         # self.model = GCN(in_channels, *hidden_channels, in_channels, generalize=True, bias=False, final_bias=False, **self.kwargs)
#         self.model = gnn.Sequential('x, edges', [
#             (GCN(in_channels, self.hidden_channels[0], bias=False, final_bias=False), 'x, edges -> x1'),
#             MLP(*self.hidden_channels, in_channels, bias=False, final_bias=False)
#         ])
#         self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
#         self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, learning_rate, n_epochs)

#         return self
    
#     def forward(self, seed, update_rate=1e-4, n_steps=1):
#         """Performs the specified number of forward passes through the network.

#         Parameters
#         ----------
#         seed : Tensor
#             TODO
#         update_rate : float, default=1e-4
#             TODO
#         n_steps : int, default=1
#             Number of forward passes through the network.

#         Returns
#         -------
#         Tensor
#             TODO
#         """
        
#         x = seed.clone()

#         for _ in range(n_steps):
#             edges = knn(x[:, :2], x[:, :2], self.n_neighbors)
#             update = self.model(x, edges)
#             x = x + update*update_rate

#         return x
    
#     def step(self, out, target):
#         loss = self.loss_function(out, target)
#         loss.backward()
#         self.optimizer.step()
#         self.scheduler.step()
#         self.optimizer.zero_grad()

#         return loss.item()
    
#     def train(self, seed, target, n_epochs=4000, learning_rate=1e-1, update_rate=1e-3, min_steps=64, verbosity=2, display_rate=1):
#         self.build(seed, n_epochs, learning_rate)
#         n_steps, max_steps = min_steps, min_steps + 1

#         for i in tqdm(range(n_epochs)) if verbosity > 0 else range(n_epochs):
#             out = self(seed, update_rate, n_steps)
#             self.loss_log.append(self.step(out[:, :5], target))

#             if i > n_epochs/2:
#                 max_steps += min_steps/n_epochs
#                 n_steps = torch.randint(min_steps, int(max_steps), (1,)).item()

#             if verbosity == 2 and i%display_rate == 0:
#                 show_progress(out.detach(), self.loss_log, f'steps: {n_steps}', f'epochs: {i + 1}')

#         return self
    
# # def ot_loss(x, y, particle_dim=2, projection_dim=128):
# #     projection = torch.normal(0., 1., (particle_dim, projection_dim))
# #     projection /= projection.square().sum(0, keepdim=True).sqrt()
# #     points = (x[:, :particle_dim]@projection).T.sort()[0]
# #     target = (y[:, :particle_dim]@projection).T.sort()[0]
# #     mask = (torch.linspace(0, target.shape[-1] - 1, x.shape[0]) + .5).long()
# #     target = target[..., mask]
# #     loss = (points - target).square().sum()/projection_dim

# #     return loss

# # class NPA(nn.Module):
# #     def __init__(self, n_neighbors=9, hidden_channels=64, loss_function=ot_loss):
# #         super().__init__()

#         self.n_neighbors = n_neighbors
#         self.hidden_channels = hidden_channels
#         self.loss_function = loss_function

#         self.network = None
#         self.optimizer = None
#         self.scheduler = None
#         self.loss_log = []

#     def build(self, seed, n_epochs=1000, learning_rate=1e-2):
#         in_channels = seed.shape[-1]
#         self.network = GENConv(in_channels, in_channels, num_layers=4)
#         self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
#         self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, learning_rate, n_epochs)

#         return self
    
#     def forward(self, x, n_steps=1, update_rate=1e-4):
#         for _ in range(n_steps):
#             edges = knn(x[:, :2], x[:, :2], self.n_neighbors)
#             update = self.network(x, edges)
#             x = x + update*update_rate

#         return x
    
#     def step(self, x, target):
#         loss = self.loss_function(x, target)
#         loss.backward()
#         self.optimizer.step()
#         self.scheduler.step()
#         self.optimizer.zero_grad()

#         return loss.item()
    
#     def display(self, x, epoch, n_steps, verbosity=2, display_rate=5, size=1, figsize=(10, 5)):
#         if verbosity == 2 and epoch%display_rate == 0:
#             clear_output(True)
#             particles = x.detach()
#             _, (subplot1, subplot2) = plt.subplots(1, 2, figsize=figsize)
#             subplot1.axis('off')
#             subplot1.axis((-5, 135, -5, 135))
#             # colors = torch.clip(particles[:, 2:5], 0, 1)
#             subplot1.set_title(f'steps: {n_steps}')
#             subplot1.scatter(*particles[:, :2].T, size)
#             subplot2.set_title(f'epochs: {epoch + 1}')
#             subplot2.plot(torch.arange(epoch + 1), self.loss_log)
#             plt.show()
    
#     def train(self, seed, target, n_epochs=4000, learning_rate=1e-1, update_rate=1e-2, min_steps=64, verbosity=2, display_rate=1):
#         self.build(seed, n_epochs, learning_rate)
#         max_steps = min_steps + 1

#         for i in tqdm(range(n_epochs)) if verbosity > 0 else range(n_epochs):
#             x = seed.clone()
#             n_steps = torch.randint(min_steps, int(max_steps), (1,)).item()
#             x = self(x, n_steps, update_rate)
#             loss = self.step(x, target)
#             self.loss_log.append(loss)
#             max_steps += (i > n_epochs/2)*min_steps/n_epochs
#             self.display(x[:, :5].detach(), i, n_steps, verbosity, display_rate)

#         return self
