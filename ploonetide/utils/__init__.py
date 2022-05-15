"""This module provides various helper functions."""
import glob
import os
import logging
import shutil
import subprocess

import functools

from natsort import natsorted
from warnings import warn

from astropy.io import fits

log = logging.getLogger(__name__)

__all__ = ['dict2obj', 'logged', 'set_xaxis_limits']


class dict2obj(object):
    def __init__(self, dic={}):
        self.__dict__.update(dic)

    def __add__(self, other):
        for attr in other.__dict__.keys():
            exec(f"self.{attr}=other.{attr}")
        return self


def set_xaxis_limits(ax, ax1):
    lim1 = ax.get_xlim()
    lim2 = ax1.get_xlim()

    return lim2[0] + (ax.get_xticks() - lim1[0]) / (lim1[1] - lim1[0]) * (lim2[1] - lim2[0])


def logged(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(func.__name__ + " was called")
        return func(*args, **kwargs)
    return wrapper
