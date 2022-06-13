"""This module provides various helper functions."""
import logging
import functools
import numpy as np
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt

from scipy.constants import G as Gconst


log = logging.getLogger(__name__)

__all__ = ['dict2obj', 'logged', 'set_xaxis_limits']


class dict2obj(object):
    def __init__(self, dic={}):
        self.__dict__.update(dic)

    def __add__(self, other):
        for attr in other.__dict__.keys():
            exec(f'self.{attr}=other.{attr}')
        return self


def colorline(x, y, z=None, cmap='copper', linewidth=2, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, '__iter__'):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, linewidth=linewidth, alpha=alpha,
                              norm=plt.Normalize(np.nanmin(z), np.nanmax(z)))

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def set_xaxis_limits(ax, ax1):
    lim1 = ax.get_xlim()
    lim2 = ax1.get_xlim()

    return lim2[0] + (ax.get_xticks() - lim1[0]) / (lim1[1] - lim1[0]) * (lim2[1] - lim2[0])


def logged(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(func.__name__ + ' was called')
        return func(*args, **kwargs)
    return wrapper


#############################################################
# CANONICAL UNITS TRANSFORMATION
#############################################################
def canonic_units(**kwargs):
    """Convert to chosen units

    Args:
        **kwargs: keyword arguments

    Returns:
        float: Unit conversion factors
    """
    G = Gconst
    if 'uM' in kwargs.keys() and 'uL' in kwargs.keys():
        uT = (kwargs['uL']**3 / (G * kwargs['uM']))**0.5
        return [kwargs.get('uM'), kwargs.get('uL'), uT]

    elif 'uM' in kwargs.keys() and 'uT' in kwargs.keys():
        uL = (G * kwargs['uM'] * kwargs['uT']**2)**(1.0 / 3.0)
        return [kwargs.get('uM'), uL, kwargs.get('uT')]

    elif 'uL' in kwargs.keys() and 'uT' in kwargs.keys():
        uM = (kwargs['uL']**3 / (kwargs['uT']**2 * G))
        return [uM, kwargs.get('uL'), kwargs.get('uT')]


# THIS IS THE FIXED-POINT FUNCTION FOR DOING THE "k" ITERATIONS TO
# REFINE THE ROOT'S VALUE.
def fpi(t, k, ll, n):
    for i in np.arange(k):
        # print(i,g(t,l,n))
        t = g(t, ll, n)
    return t


# ############################################################
# Util Functions
# ############################################################
def fmt(x, pos):
    """Write scientific notation formater for plots.

    Args:
        x (int): Description
        pos (TYPE): Description

    Returns:
        string: Scientific notation of input string
    """
    a, b = f'{x:.1e}'.split('e')
    b = int(b)
    return rf'${a} \times 10^{{{b}}}$'
