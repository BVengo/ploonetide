#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

from ploonetide.utils.functions import *
from ploonetide.utils.constants import GR, GCONST


#############################################################
# DIFFERENTIAL EQUATIONS
#############################################################
def dnmdt(q, t, sim_parameters):
    """Define the differential equation for the moon mean motion.

    Args:
        q (list): vector defining nm
        t (float): time
        sim_parameters (dict): Dictionary that contains all the sim_parameters for the ODEs..

    Returns:
        list: Rate of change of the moon mean motion
    """
    nm = q[0]

    # Evolving conditions
    args = sim_parameters['args']

    # Primary properties
    Mp = sim_parameters['Mp']
    alpha_planet = sim_parameters['planet_alpha']
    beta_planet = sim_parameters['planet_beta']
    rigidity = sim_parameters['rigidity']
    Mm = sim_parameters['Mm']

    # Dynamic parameter
    op = sim_parameters['op']
    if sim_parameters['em_ini'] == 0.0:
        eccm = 0.0
    else:
        eccm = sim_parameters['eccm']

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

    if sim_parameters['em_ini'] == 0.0:
        dnmdt = (-9. / 2 * k2q_planet * Mm * Rp**5 / (GCONST**(5. / 3) * Mp**(8. / 3))
                 * nm**(16. / 3) * np.sign(op - nm))
    else:
        dnmdt = 9. * nm**(16. / 3.) * k2q_planet * Mm * Rp**5. / (Mp * (GCONST * (Mp + Mm))**(5. / 3.)) *\
            ((1. + 23. * eccm**2.) - (1. + 13.5 * eccm**2.) * op / nm)

    return [dnmdt]


def demdt(q, t, sim_parameters):
    """Define the differential equation for the eccentricity of the moon.

    Args:
        q (list): vector defining em
        t (float): time
        sim_parameters (dict): Dictionary that contains all the sim_parameters for the ODEs..

    Returns:
        list: Eccentricity of the moon
    """
    eccm = q[0]

    # Evolving conditions
    args = sim_parameters['args']

    # Primary properties
    Mp = sim_parameters['Mp']
    alpha_planet = sim_parameters['planet_alpha']
    beta_planet = sim_parameters['planet_beta']
    rigidity = sim_parameters['rigidity']
    Mm = sim_parameters['Mm']

    # Dynamic parameter
    op = sim_parameters['op']
    nm = sim_parameters['nm']

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


def dopdt(q, t, sim_parameters):
    """Define the differential equation for the rotational rate of the planet.

    Args:
        q (list): vector defining op
        t (float): time
        sim_parameters (dict): Dictionary that contains all the sim_parameters for the ODEs.

    Returns:
        list: rotational rate of the planet
    """
    op = q[0]

    # Evolving conditions
    args = sim_parameters['args']

    # Primary properties
    Mp = sim_parameters['Mp']
    alpha_planet = sim_parameters['planet_alpha']
    beta_planet = sim_parameters['planet_beta']
    rigidity = sim_parameters['rigidity']
    Mm = sim_parameters['Mm']
    npp = sim_parameters['npp']

    # Dynamic parameter
    nm = sim_parameters['nm']
    npp = sim_parameters['npp']

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


def dnpdt(q, t, sim_parameters):
    """Define the differential equation for the mean motion of the planet.

    Args:
        q (list): vector defining np
        t (float): time
        sim_parameters (dict): Dictionary that contains all the sim_parameters for the ODEs.

    Returns:
        list: mean motion of the planet
    """
    npp = q[0]

    # Evolving conditions
    args = sim_parameters['args']

    # Primary properties
    Ms = sim_parameters['Ms']
    Mp = sim_parameters['Mp']
    alpha_planet = sim_parameters['planet_alpha']
    beta_planet = sim_parameters['planet_beta']
    rigidity = sim_parameters['rigidity']

    # Dynamic parameter
    op = sim_parameters['op']

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
def dEmdt(q, t, sim_parameters):
    """Define the differential equation for the tidal energy of the moon.

    Args:
        q (list): vector defining Em
        t (float): time
        sim_parameters (dict): Dictionary that contains all the sim_parameters for the ODEs.

    Returns:
        list: Tidal energy of the moon
    """
    E = q[0]

    # General sim_parameters
    E_act = sim_parameters['E_act']
    B = sim_parameters['B']
    Ts = sim_parameters['Ts']
    Tb = sim_parameters['Tb']
    Tl = sim_parameters['Tl']

    # Moon sim_parameters
    densm = sim_parameters['densm']
    Mm = sim_parameters['Mm']
    Rm = sim_parameters['Rm']
    melt_fr = sim_parameters['melt_fr']

    # Dynamic sim_parameters
    Tm = sim_parameters['Tm']
    nm = sim_parameters['nm']

    if sim_parameters['em_ini'] == 0.0:
        eccm = 0.0
    else:
        eccm = sim_parameters['eccm']

    dEdt = e_tidal(Tm, nm, densm=densm, Mm=Mm, Rm=Rm, E_act=E_act, melt_fr=melt_fr, B=B, Ts=Ts,
                   Tb=Tb, Tl=Tl, eccm=eccm)

    return [dEdt]


#############################################################
# INTEGRATION OF THE TEMPERATURE
#############################################################
def dTmdt(q, t, sim_parameters):
    """Define the differential equation for the temperatue of the moon.

    Args:
        q (list): vector defining Tm
        t (float): time
        sim_parameters (dict): Dictionary that contains all the sim_parameters for the ODEs.

    Returns:
        list: Temperature of the moon
    """
    Tm = q[0]

    # General sim_parameters
    E_act = sim_parameters['E_act']
    B = sim_parameters['B']
    Ts = sim_parameters['Ts']
    Tb = sim_parameters['Tb']
    Tl = sim_parameters['Tl']
    Cp = sim_parameters['Cp']
    ktherm = sim_parameters['ktherm']
    Rac = sim_parameters['Rac']
    a2 = sim_parameters['a2']
    alpha_exp = sim_parameters['alpha_exp']

    # Moon sim_parameters
    densm = sim_parameters['densm']
    Mm = sim_parameters['Mm']
    Rm = sim_parameters['Rm']
    melt_fr = sim_parameters['melt_fr']

    # Dynamic parameter
    nm = sim_parameters['nm']

    if sim_parameters['em_ini'] == 0.0:
        eccm = 0.0
    else:
        eccm = sim_parameters['eccm']

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
def solution_planet_moon(q, t, sim_parameters):
    """Define the coupled differential equation for the system of EDOs.

    Args:
        q (list): vector defining np
        t (float): time
        sim_parameters (dict): Dictionary that contains all the sim_parameters for the ODEs.

    Returns:
        list: mean motion of the planet
    """
    nm = q[0]
    op = q[1]
    npp = q[2]
    Tm = q[3]
    Em = q[4]

    if sim_parameters['em_ini'] != 0.0:
        eccm = q[5]
        sim_parameters['eccm'] = eccm

    sim_parameters['nm'] = nm
    sim_parameters['op'] = op
    sim_parameters['npp'] = npp
    sim_parameters['Tm'] = Tm
    sim_parameters['Em'] = Em

    dnmdtm = dnmdt([nm], t, sim_parameters)
    dopdtp = dopdt([op], t, sim_parameters)
    dnpdtp = dnpdt([npp], t, sim_parameters)
    dTmdtm = dTmdt([Tm], t, sim_parameters)
    dEmdtm = dEmdt([Em], t, sim_parameters)

    solution = dnmdtm + dopdtp + dnpdtp + dTmdtm + dEmdtm

    if sim_parameters['em_ini'] != 0.0:
        demdtm = demdt([eccm], t, sim_parameters)
        solution = dnmdtm + dopdtp + dnpdtp + dTmdtm + dEmdtm + demdtm

    return solution
