import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import clear_output, display, Video

def get_particles(path, threshold=.1, n_channels=0):
    image = torch.tensor(plt.imread(path))
    points = torch.stack(torch.where(image[..., -1] >= threshold)[::-1])
    colors = image[points[1], points[0], :3]
    points[1] = points[1].max() - points[1]
    particles = torch.cat([points.T, colors, torch.ones((points.shape[-1], n_channels))], dim=-1)

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
    frame_rage : float, default=30.0
        Video frame rate.
    file_name : str, default='_autoplay.mp4'
        Video file name.

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

    def __init__(self, size=500, frame_rate=30., file_name='_autoplay.mp4'):
        self.size = size
        self.frame_rate = frame_rate
        self.file_name = file_name

        self.frames = []

    def __enter__(self):
        return self
    
    def __exit__(self, *_):
        self.save()

        if self.file_name == '_autoplay.mp4':
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

        with imageio.imopen(self.file_name, 'w', plugin='pyav') as out:
            out.init_video_stream('vp9', fps=self.frame_rate)

            for f in self.frames:
                out.write_frame(f)

    def show(self):
        video = Video(self.file_name, width=self.size, height=self.size)
        display(video)
