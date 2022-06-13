#!/usr/bin/env python
# -*- coding:utf-8 -*-
#############################################################
# EXTERNAL PACKAGES
#############################################################
from scipy import constants as const
from astropy import constants as const_as

from ploonetide.utils import dict2obj

# ############################################################
# CONSTANTS
# ############################################################
# Exponent and coefficients of radius mass-scaling law
ALPHAR = 0.156
A = 3.964
B = -0.064
C = 3.364
t0 = 1e8

# Scaling constants for alpha and beta
KP = 90.0742985384
BP = 4.0
DP = -0.232

#############################################################
# GYRATION RADIUS
#############################################################
GR = 0.2  # None

#############################################################
# CONSTANTS
#############################################################

# PHYSICAL CONSTANTS
MIN = const.minute  # s
HOUR = const.hour  # s
DAY = const.day  # s
YEAR = const.Julian_year  # s
KYEAR = const.kilo * YEAR  # s
MYEAR = const.mega * YEAR  # s
GYEAR = const.giga * YEAR  # s
GCONST = const.G  # m^3 / kg s^2

# General constants
gas_constant = const.gas_constant  # J mol^-1 K^-1  -->  kg * m^2 s^-2 * mol^-1 * K-1
stefan_b_constant = const.sigma  # W m^-2 K^-4  -->  kg s^-3 K^-4

# ASTRONOMICAL CONSTANTS
AU = const.au  # m
MSUN = const_as.M_sun.value  # kg
RSUN = const_as.R_sun.value  # m
LSUN = const_as.L_sun.value  # W
MJUP = const_as.M_jup.value
MENCEL = 1.08e20  # kg
MTITAN = 1.345e23  # kg
MEARTH = const_as.M_earth.value  # kg
REARTH = const_as.R_earth.value  # m

PLANETS = dict2obj(dict(
    Jupiter=dict2obj(dict(M=1.898e27,  # kg
                          R=6.9911e7,  # m
                          P=29.5 * YEAR,  # s
                          Prot=9.4 * HOUR,  # s
                          alpha=0.126,
                          beta=0.020)),
    Saturn=dict2obj(dict(M=5.683e26,  # kg
                         R=6.0268e7,  # m
                         P=10.8 * YEAR,  # s
                         Prot=10.656 * HOUR,  # s
                         alpha=0.219,
                         beta=0.196)),
    Uranus=dict2obj(dict(M=86.8e24,  # kg
                         R=2.5632e7,  # m
                         P=84 * YEAR,  # s
                         Prot=17.24 * HOUR,  # s
                         alpha=0.30,
                         beta=0.093)),
    Neptune=dict2obj(dict(M=1.024e26,  # kg
                          R=2.4622e7,  # m
                          P=164.8 * YEAR,  # s
                          Prot=16.11 * HOUR,  # s
                          alpha=0.35,
                          beta=0.131,))))
