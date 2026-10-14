"""
Authors: Craig Fouts
License: Apache 2.0 license
"""

import matplotlib.pyplot as plt
import torch

def load_pts(path, n_channels=0, fill=0., cutoff=1e-2):
    img = torch.tensor(plt.imread(path))
    locs = torch.stack(torch.where(img[..., -1] >= cutoff)[::-1])
    clrs = img[locs[1], locs[0], :3]
    pts = torch.cat([locs.T, clrs, fill*torch.ones(locs.shape[1], n_channels)], 1)

    return pts
