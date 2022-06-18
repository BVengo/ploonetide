#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

from ploonetide.utils.functions import *
from ploonetide.utils.constants import GR, GCONST


#############################################################
# DIFFERENTIAL EQUATIONS
#############################################################
def dnmdt(q, t, parameters):
    """Define the differential equation for the moon mean motion.

    Args:
        q (list): vector defining nm
        t (float): time
        parameters (dict): Dictionary that contains all the parameters for the ODEs..

    Returns:
        list: Rate of change of the moon mean motion
    """
    nm = q[0]

    # Evolving conditions
    args = parameters['args']

    # Primary properties
    Mp = parameters['Mp']
    alpha_planet = parameters['planet_alpha']
    beta_planet = parameters['planet_beta']
    rigidity = parameters['rigidity']
    Mm = parameters['Mm']

    # Dynamic parameter
    op = parameters['op']
    if parameters['em_ini'] == 0.0:
        eccm = 0.0
    else:
        eccm = parameters['eccm']

    # Secondary properties
    if not args['planet_size_evolution']:
        Rp = args['Rp']
    else:
        Rp = Mp2Rp(Mp, t)
        alpha_planet = alpha_planet * args['Rp'] / Rp

    epsilon = op / omegaCritic(Mp, Rp)
    # beta=alpha2beta(Mp,alpha,**args)
    if not args['planet_internal_evolution']:
        k2q_planet = args['planet_k2q']
    else:
        k2q_planet_core = 0.0
        if args['planet_core_dissipation']:
            k2q_planet_core = k2Q_planet_core(rigidity, alpha_planet, beta_planet, Mp, Rp)
        k2q_planet_envelope = k2Q_planet_envelope(alpha_planet, beta_planet, epsilon)
        k2q_planet = k2q_planet_core + k2q_planet_envelope

    if parameters['em_ini'] == 0.0:
        dnmdt = (-9. / 2 * k2q_planet * Mm * Rp**5 / (GCONST**(5. / 3) * Mp**(8. / 3))
                 * nm**(16. / 3) * np.sign(op - nm))
    else:
        dnmdt = 9. * nm**(16. / 3.) * k2q_planet * Mm * Rp**5. / (Mp * (GCONST * (Mp + Mm))**(5. / 3.)) *\
            ((1. + 23. * eccm**2.) - (1. + 13.5 * eccm**2.) * op / nm)

    return [dnmdt]


def demdt(q, t, parameters):
    """Define the differential equation for the eccentricity of the moon.

    Args:
        q (list): vector defining em
        t (float): time
        parameters (dict): Dictionary that contains all the parameters for the ODEs..

    Returns:
        list: Eccentricity of the moon
    """
    eccm = q[0]

    # Evolving conditions
    args = parameters['args']

    # Primary properties
    Mp = parameters['Mp']
    alpha_planet = parameters['planet_alpha']
    beta_planet = parameters['planet_beta']
    rigidity = parameters['rigidity']
    Mm = parameters['Mm']

    # Dynamic parameter
    op = parameters['op']
    nm = parameters['nm']

    # Secondary properties
    if not args['planet_size_evolution']:
        Rp = args['Rp']
    else:
        Rp = Mp2Rp(Mp, t)
        alpha_planet = alpha_planet * args['Rp'] / Rp

    epsilon = op / omegaCritic(Mp, Rp)
    # beta=alpha2beta(Mp,alpha,**args)
    if not args['planet_internal_evolution']:
        k2q_planet = args['planet_k2q']
    else:
        k2q_planet_core = 0.0
        if args['planet_core_dissipation']:
            k2q_planet_core = k2Q_planet_core(rigidity, alpha_planet, beta_planet, Mp, Rp)
        k2q_planet_envelope = k2Q_planet_envelope(alpha_planet, beta_planet, epsilon)
        k2q_planet = k2q_planet_core + k2q_planet_envelope

    demdt = -27. * nm**(13. / 3.) * eccm * k2q_planet * Mm * Rp**5. \
        / (Mp * (GCONST * (Mp + Mm))**(5. / 3.)) * (1. - 11. / 18. * op / nm)

    return [demdt]


def dopdt(q, t, parameters):
    """Define the differential equation for the rotational rate of the planet.

    Args:
        q (list): vector defining op
        t (float): time
        parameters (dict): Dictionary that contains all the parameters for the ODEs.

    Returns:
        list: rotational rate of the planet
    """
    op = q[0]

    # Evolving conditions
    args = parameters['args']

    # Primary properties
    Mp = parameters['Mp']
    alpha_planet = parameters['planet_alpha']
    beta_planet = parameters['planet_beta']
    rigidity = parameters['rigidity']
    Mm = parameters['Mm']
    npp = parameters['npp']

    # Dynamic parameter
    nm = parameters['nm']
    npp = parameters['npp']

    # Secondary properties
    if not args['planet_size_evolution']:
        Rp = args['Rp']
    else:
        Rp = Mp2Rp(Mp, t, **args)
        alpha_planet = alpha_planet * args['Rp'] / Rp

    epsilon = op / omegaCritic(Mp, Rp)
    # beta=alpha2beta(Mp,alpha,**args)
    if args['planet_internal_evolution']:
        k2q_planet = args['planet_k2q']
    else:
        k2q_planet_core = 0.0
        if args['planet_core_dissipation']:
            k2q_planet_core = k2Q_planet_core(rigidity, alpha_planet, beta_planet, Mp, Rp)
        k2q_planet_envelope = k2Q_planet_envelope(alpha_planet, beta_planet, epsilon)
        k2q_planet = k2q_planet_core + k2q_planet_envelope

    dopdt = -3. / 2. * k2q_planet * Rp**3 / (GR * GCONST) *\
        (Mm**2. * nm**4. * np.sign(op - nm) / Mp**3 + npp**4. * np.sign(op - npp) / Mp)

    # dopdt = -3. / 2. * k2q * Rp**3 / (GR * GCONST) *\
    #     (Mm**2. * nm**4. * np.sign(op - nm) / Mp**3
    #      + (GCONST * Ms)**2. * np.sign(op - nmp) / (Mp * ap**6.))

    return [dopdt]


def dnpdt(q, t, parameters):
    """Define the differential equation for the mean motion of the planet.

    Args:
        q (list): vector defining np
        t (float): time
        parameters (dict): Dictionary that contains all the parameters for the ODEs.

    Returns:
        list: mean motion of the planet
    """
    npp = q[0]

    # Evolving conditions
    args = parameters['args']

    # Primary properties
    Ms = parameters['Ms']
    Mp = parameters['Mp']
    alpha_planet = parameters['planet_alpha']
    beta_planet = parameters['planet_beta']
    rigidity = parameters['rigidity']

    # Dynamic parameter
    op = parameters['op']

    # Secondary properties
    if not args['planet_size_evolution']:
        Rp = args['Rp']
    else:
        Rp = Mp2Rp(Mp, t, **args)
        alpha_planet = alpha_planet * args['Rp'] / Rp

    epsilon = op / omegaCritic(Mp, Rp)
    # beta=alpha2beta(Mp,alpha,**args)
    if not args['planet_internal_evolution']:
        k2q_planet = args['planet_k2q']
    else:
        k2q_planet_core = 0.0
        if args['planet_core_dissipation']:
            k2q_planet_core = k2Q_planet_core(rigidity, alpha_planet, beta_planet, Mp, Rp)
        k2q_planet_envelope = k2Q_planet_envelope(alpha_planet, beta_planet, epsilon)
        k2q_planet = k2q_planet_core + k2q_planet_envelope

    dnpdt = (-9. / 2 * k2q_planet * Rp**5 / (GCONST**(5. / 3.) * Mp * Ms**(2. / 3.))
             * npp**(16. / 3) * np.sign(op - npp))

    return [dnpdt]


#############################################################
# INTEGRATION OF THE TIDAL HEAT
#############################################################
def dEmdt(q, t, parameters):
    """Define the differential equation for the tidal energy of the moon.

    Args:
        q (list): vector defining Em
        t (float): time
        parameters (dict): Dictionary that contains all the parameters for the ODEs.

    Returns:
        list: Tidal energy of the moon
    """
    E = q[0]

    # General parameters
    E_act = parameters['E_act']
    B = parameters['B']
    Ts = parameters['Ts']
    Tb = parameters['Tb']
    Tl = parameters['Tl']

    # Moon parameters
    densm = parameters['densm']
    Mm = parameters['Mm']
    Rm = parameters['Rm']
    melt_fr = parameters['melt_fr']

    # Dynamic parameters
    Tm = parameters['Tm']
    nm = parameters['nm']

    if parameters['em_ini'] == 0.0:
        eccm = 0.0
    else:
        eccm = parameters['eccm']

    dEdt = e_tidal(Tm, nm, densm=densm, Mm=Mm, Rm=Rm, E_act=E_act, melt_fr=melt_fr, B=B, Ts=Ts,
                   Tb=Tb, Tl=Tl, eccm=eccm)

    return [dEdt]


#############################################################
# INTEGRATION OF THE TEMPERATURE
#############################################################
def dTmdt(q, t, parameters):
    """Define the differential equation for the temperatue of the moon.

    Args:
        q (list): vector defining Tm
        t (float): time
        parameters (dict): Dictionary that contains all the parameters for the ODEs.

    Returns:
        list: Temperature of the moon
    """
    Tm = q[0]

    # General parameters
    E_act = parameters['E_act']
    B = parameters['B']
    Ts = parameters['Ts']
    Tb = parameters['Tb']
    Tl = parameters['Tl']
    Cp = parameters['Cp']
    ktherm = parameters['ktherm']
    Rac = parameters['Rac']
    a2 = parameters['a2']
    alpha_exp = parameters['alpha_exp']

    # Moon parameters
    densm = parameters['densm']
    Mm = parameters['Mm']
    Rm = parameters['Rm']
    melt_fr = parameters['melt_fr']

    # Dynamic parameter
    nm = parameters['nm']

    if parameters['em_ini'] == 0.0:
        eccm = 0.0
    else:
        eccm = parameters['eccm']

    dEdt = e_tidal(Tm, nm, densm=densm, Mm=Mm, Rm=Rm, E_act=E_act, melt_fr=melt_fr,
                   B=B, Ts=Ts, Tb=Tb, Tl=Tl, eccm=eccm)

    if Tm < Ts:
        eta = eta_below_Ts(Tm, E_act=E_act)

    elif Ts <= Tm < Tb:
        eta = eta_between_Ts_Tb(Tm, E_act=E_act, melt_fr=melt_fr, B=B)

    elif Tb <= Tm < Tl:
        eta = eta_between_Tb_Tl(Tm, melt_fr=melt_fr)

    else:
        eta = eta_above_Tl(Tm)

    # Calculation of convection
    kappa = ktherm / (densm * Cp)

    C = Rac**0.25 / (2 * a2) * (alpha_exp * gravity(Mm, Rm) * densm
                                / (eta * kappa * ktherm))**-0.25
    qBL = (ktherm * (Tm - surf_temp(dEdt, Rm)) / C)**(4. / 3.)

    # qBL = ktherm / 2. * (densm * gravity(Mm, Rm) * alpha_exp / (kappa * eta))**(1. / 3.) *\
    #     (E_act / (Rg * Tm**2.))**(-4. / 3.)

    coeff = 4. / 3. * np.pi * (Rm**3. - (0.4 * Rm)**3.) * densm * Cp

    dTdt = (-qBL + dEdt) / coeff

    return [dTdt]


#############################################################
# INTEGRATION OF THE WHOLE SYSTEM
#############################################################
def solution_planet_moon(q, t, parameters):
    """Define the coupled differential equation for the system of EDOs.

    Args:
        q (list): vector defining np
        t (float): time
        parameters (dict): Dictionary that contains all the parameters for the ODEs.

    Returns:
        list: mean motion of the planet
    """
    nm = q[0]
    op = q[1]
    npp = q[2]
    Tm = q[3]
    Em = q[4]

    if parameters['em_ini'] != 0.0:
        eccm = q[5]
        parameters['eccm'] = eccm

    parameters['nm'] = nm
    parameters['op'] = op
    parameters['npp'] = npp
    parameters['Tm'] = Tm
    parameters['Em'] = Em

    dnmdtm = dnmdt([nm], t, parameters)
    dopdtp = dopdt([op], t, parameters)
    dnpdtp = dnpdt([npp], t, parameters)
    dTmdtm = dTmdt([Tm], t, parameters)
    dEmdtm = dEmdt([Em], t, parameters)

    solution = dnmdtm + dopdtp + dnpdtp + dTmdtm + dEmdtm

    if parameters['em_ini'] != 0.0:
        demdtm = demdt([eccm], t, parameters)
        solution = dnmdtm + dopdtp + dnpdtp + dTmdtm + dEmdtm + demdtm

    return solution
