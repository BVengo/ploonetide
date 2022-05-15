"""Defines TimeSeriesData"""

from __future__ import division
import os
import warnings
import logging

import pandas as pd
import numpy as np
import numpy.ma as ma
import time
import pyfiglet
import scipy

from . import PACKAGEDIR
from ploonetide.forecaster import mr_forecast

from ploonetide.utils import (
    search_files_across_directories,
    catalog,
    get_data,
    get_header
)

from ploonetide.noise import (
    compute_scintillation,
    compute_noises
)

__all__ = ['TimeSeriesAnalysis', 'LightCurve']


warnings.simplefilter('ignore', category=AstropyWarning)
