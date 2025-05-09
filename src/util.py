"""
Craig Fouts (craig.fouts@uu.igp.se)
"""

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display, Video

def transport_loss(x, y, projection_dim=128, scale=1.):
    projection = torch.normal(0., 1., (x.shape[-1], projection_dim))
    projection /= projection.square().sum(0, keepdim=True).sqrt()
    particles = (x@projection).T.sort()[0]
    target = (y@projection).T.sort()[0]
    mask = (torch.linspace(0, target.shape[-1] - 1, x.shape[0]) + .5).long()
    target = target[:, mask]
    loss = scale*(particles - target).square().sum()/projection_dim

    return loss

def get_particles(path, threshold=.5, n_channels=0, randomize=False):
    image = torch.tensor(plt.imread(path))
    points = torch.stack(torch.where(image[..., -1] >= threshold)[::-1])
    colors = image[points[1], points[0], :3]
    points[1] = points[1].max() - points[1]

    if randomize:
        particles = torch.cat([points.T, colors, torch.rand(points.shape[-1], n_channels)], dim=-1)
    else:
        particles = torch.cat([points.T, colors, torch.zeros(points.shape[-1], n_channels)], dim=-1)

    return particles

def show_particles(*particles, size=1, limits=(0, 125, 0, 125), clear=False, show=True):
    plt.axis('equal')
    plt.axis('off')
    plt.axis(limits)

    for p in particles:
        colors = torch.clip(p[:, 2:5], 0, 1) if p.shape[-1] > 2 else None
        plt.scatter(*p[:, :2].T, size, colors)

    if clear:
        clear_output()
    
    if show:
        plt.show()

def show_progress(particles, log, particles_title=None, log_title=None, particles_size=1, particles_range=(-5, 135, -5, 135), clear=True):
    if clear:
        clear_output(True)

    _, (particles_plot, log_plot) = plt.subplots(1, 2, figsize=(10, 5))
    particles_plot.axis('off')
    particles_plot.axis(particles_range)
    colors = torch.clip(particles[:, 2:5], 0, 1) if particles.shape[-1] > 2 else torch.zeros(particles.shape[0])
    particles_plot.set_title(particles_title)
    particles_plot.scatter(*particles[:, :2].T, particles_size, colors)
    log_plot.set_title(log_title)
    log_plot.plot(torch.arange(len(log)), log)
    plt.show()

def grab_plot(close=True):
    """Converts the current Matplotlib canvas into an RGB image array.
    
    Parameters
    ----------
    close : bool, default=True
        Whether to close the plot after conversion.

    Returns
    -------
    ndarray
        RGB image array.
    """

    figure = plt.gcf()
    figure.canvas.draw()
    image = np.array(figure.canvas.renderer._renderer)
    alpha = np.float32(image[..., 3:]/255.)
    image = np.uint8(255.*(1. - alpha) + image[..., :3]*alpha)

    if close:
        plt.close()

    return image

class VideoWriter:
    """Utilitiy that writes RGB image arrays into a video stream.
    
    Parameters
    ----------
    size : int, default=500
        Video canvas width and height.
    rate : float, default=30.0
        Video frame rate.
    path : str, default='_autoplay.mp4'
        Video file path.

    Attributes
    ----------
    frames : list
        Sequence of RGB image arrays.

    Usage
    -----
    >>> with VideoWriter(frame_rate=15) as video:
    >>>     for i in range(1, 100):
    >>>         x = np.arange(i)
    >>>         plt.plot(x, np.sin(x))
    >>>         video.write(grab_plot())
    """

    def __init__(self, size=500, rate=30., path='../videos/_autoplay.mp4'):
        self.size = size
        self.rate = rate
        self.path = path

        self.frames = []

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.save()

        if self.path[-13:] == '_autoplay.mp4':
            self.show()

    def write(self, image):
        """Adds the given RGB image array to the frame sequence.
        
        Parameters
        ----------
        image : ndarray
            RGB image array.

        Returns
        -------
        None
        """

        self.frames.append(image)

    def save(self):
        """Converts the current frame sequence into a video stream.
        
        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        with imageio.imopen(self.path, 'w', plugin='pyav') as out:
            out.init_video_stream('vp9', fps=self.rate)

            for f in self.frames:
                out.write_frame(f)

    def show(self):
        video = Video(self.path, width=self.size, height=self.size)
        display(video)
