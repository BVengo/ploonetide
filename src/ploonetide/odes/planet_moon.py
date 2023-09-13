#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

from ploonetide.utils.functions import *
from ploonetide.utils.constants import GR, GCONST


#############################################################
# DIFFERENTIAL EQUATIONS
#############################################################
def dnmdt(t, q, parameters):
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


def demdt(t, q, parameters):
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


def dopdt(t, q, parameters):
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


def dnpdt(t, q, parameters):
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
# INTEGRATION OF THE WHOLE SYSTEM
#############################################################
def solution_planet_moon(t, q, parameters):
    """Define the coupled differential equation for the system of EDOs.

    Args:
        q (list): vector defining np
        t (float): time
        parameters (dict): Dictionary that contains all the parameters for the ODEs.

    Returns:
        list: mean motion of the planet
    """
    op = q[0]
    npp = q[1]
    nm = q[2]

    if parameters['em_ini'] != 0.0:
        eccm = q[3]
        parameters['eccm'] = eccm

    parameters['nm'] = nm
    parameters['op'] = op
    parameters['npp'] = npp

    dopdtp = dopdt(t, [op], parameters)
    dnpdtp = dnpdt(t, [npp], parameters)
    dnmdtm = dnmdt(t, [nm], parameters)

    solution = dopdtp + dnpdtp + dnmdtm

    if parameters['em_ini'] != 0.0:
        demdtm = demdt(t, [eccm], parameters)
        solution = dopdtp + dnpdtp + dnmdtm + demdtm

    return solution
