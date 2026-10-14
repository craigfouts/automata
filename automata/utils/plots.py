"""
Authors: Craig Fouts
License: Apache 2.0 license
"""

import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output

def plot_pts(*x, size=1., width=5., height=5., color='white', clear=False, show=True, return_plot=False):
    fig, ax = plt.subplots(1, 1, figsize=(width, height), facecolor=color)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.yaxis.set_inverted(True)
    ax.set_axis_off()

    for pts in x:
        clrs = torch.clip(pts[:, 2:5], 0, 1) if pts.shape[1] > 2 else None 
        ax.scatter(*pts[:, :2].T, 2.*size**2., clrs)

    if clear:
        clear_output()

    if show:
        fig.tight_layout()

    if return_plot:
        return fig, ax

def show_log(log, x=None):
    ...
