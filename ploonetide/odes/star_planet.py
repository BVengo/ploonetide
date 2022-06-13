#!/usr/bin/env python
# -*- coding:utf-8 -*-
from ploonetide.utils.functions import *
from ploonetide.utils.constants import GCONST


#############################################################
# DIFFERENTIAL EQUATIONS
#############################################################
def depdt(q, t, parameters):
    """Define the differential equation for the eccentricity of the planet.

    Args:
        q (list): vector defining ep
        t (float): time
        parameters (dict): Dictionary that contains all the parameters for the ODEs.

    Returns:
        list: eccentricity of the planet.
    """
    e = q[0]

    # Primary properties
    Ms = parameters['Ms']
    Rs = parameters['Rs']
    alpha_planet = parameters['planet_alpha']
    beta_planet = parameters['planet_beta']
    rigidity = parameters['rigidity']
    alpha_star = parameters['star_alpha']
    beta_star_ini = parameters['star_beta']
    sun_mass_loss_rate = parameters['sun_mass_loss_rate']
    sun_omega = parameters['sun_omega']
    args = parameters['args']

    # Dynamic parameters planet
    npp = parameters['npp']
    om = parameters['om']
    os = parameters['os']
    Mp = parameters['mp']

    # Secondary properties planet
    # Rp = Mp2Rp(Mp, mpo, rpo)
    Rp = Mp2Rp(Mp, t, **args)
    epsilon_planet = om / omegaCritic(Mp, Rp)
    alpha_planet = alpha_planet * parameters['Rp'] / Rp
    beta_planet = beta_planet * parameters['Mp'] / Mp

    if not args['planet_internal_evolution']:
        k2q_planet = args['planet_k2q']
    else:
        k2q_planet_envelope = k2Q_planet_envelope(alpha_planet, beta_planet, epsilon_planet)
        k2q_planet_core = k2Q_planet_core(rigidity, alpha_planet, beta_planet, Mp, Rp)
        k2q_planet = k2q_planet_envelope + k2q_planet_core

    # Secondary properties star
    epsilon_star = os / omegaCritic(Ms, Rs)

    if not args['star_internal_evolution']:
        k2q_star = args['star_k2q']
    else:
        Ms_ini = Ms
        Ms = Ms - mloss_star(Rs, Ms, os, sun_mass_loss_rate, sun_omega) * t
        beta_star = beta_star_ini * Ms_ini / Ms
        k2q_star = k2Q_star_envelope(alpha_star, beta_star, epsilon_star, **args)

    f1ee = f1e(e)
    f2ee = f2e(e)

    ap = mean2axis(npp, Ms, Mp)
    front_term = 27 * npp * e / ap**5.0
    first_term = k2q_planet * (Ms / Mp) * Rp**5 * (11 / 18 * f2ee * om / npp - f1ee)
    second_term = k2q_star * (Mp / Ms) * Rs**5 * (11 / 18 * f2ee * os / npp - f1ee)

    depdt = front_term * (first_term + second_term)

    return [depdt]


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

    # Primary properties
    Ms = parameters['Ms']
    Rs = parameters['Rs']
    # t_star = parameters["t_star"]
    os_ini = parameters['os_ini']
    star_age = parameters['star_age']
    os_saturation = parameters['os_saturation']
    alpha_planet = parameters['planet_alpha']
    beta_planet = parameters['planet_beta']
    rigidity = parameters['rigidity']
    coeff_planet = parameters['coeff_planet']
    alpha_star = parameters['star_alpha']
    beta_star_ini = parameters['star_beta']
    coeff_star = parameters['coeff_star']
    sun_mass_loss_rate = parameters['sun_mass_loss_rate']
    sun_omega = parameters['sun_omega']
    args = parameters['args']

    # Dynamic parameters planet
    om = parameters['om']
    e = parameters["e"]
    os = parameters['os']
    Mp = parameters['mp']

    # Secondary properties planet
    # Rp = Mp2Rp(Mp, mpo, rpo)
    Rp = Mp2Rp(Mp, t, **args)

    epsilon_planet = om / omegaCritic(Mp, Rp)
    alpha_planet = alpha_planet * parameters['Rp'] / Rp
    beta_planet = beta_planet * parameters['Mp'] / Mp

    if not args['planet_internal_evolution']:
        k2q_planet = args['planet_k2q']
    else:
        k2q_planet_envelope = k2Q_planet_envelope(alpha_planet, beta_planet, epsilon_planet)
        k2q_planet_core = k2Q_planet_core(rigidity, alpha_planet, beta_planet, Mp, Rp)
        k2q_planet = k2q_planet_envelope + k2q_planet_core

    # Secondary properties star
    epsilon_star = os / omegaCritic(Ms, Rs)

    kappa = kappa_braking(os, star_age)
    osdt_braking = omegadt_braking(kappa, os, os_saturation, os_ini)

    if not args['star_internal_evolution']:
        k2q_star = args['star_k2q']
    else:
        Ms_ini = Ms
        Ms = Ms - mloss_star(Rs, Ms, os, sun_mass_loss_rate, sun_omega) * t
        beta_star = beta_star_ini * Ms_ini / Ms
        k2q_star = k2Q_star_envelope(alpha_star, beta_star, epsilon_star, **args)

    ap = mean2axis(npp, Ms, Mp)

    Ip = coeff_planet * Mp * Rp**2.0
    Is = coeff_star * Ms * Rs**2.0

    Lorb = Mp * Ms * (GCONST * ap * (1 - e**2) / (Ms + Mp))**0.5

    # Eccentricity polinomials
    f1ee = f1e(e)
    f2ee = f2e(e)
    f3ee = f3e(e)
    f4ee = f4e(e)

    front_term = 27 * npp * e / ap**5.0
    first_term = k2q_planet * (Ms / Mp) * Rp**5 * (11 / 18 * f2ee * om / npp - f1ee)
    second_term = k2q_star * (Mp / Ms) * Rs**5 * (11 / 18 * f2ee * os / npp - f1ee)

    e_p = front_term * (first_term + second_term)

    # Dobbs-Dixon 2004
    domsdt = ((3.0 * npp**4.0 * k2q_star * Rs**3.0 * Mp**2.0
               / (coeff_star * GCONST * Ms * Ms**2.0))
              * (f3ee - f4ee * os / npp))

    dompdt = ((3.0 * npp**4.0 * k2q_planet * Rp**3.0 / (coeff_planet * GCONST * Mp))
              * (f3ee - (f4ee * om / npp)))

    dnpdt = -3 * npp * (e_p * e**2.0 / (e - e**3.0) - Is * domsdt / Lorb - Ip * dompdt / Lorb + Is * osdt_braking / Lorb)

    return [dnpdt]


def dompdt(q, t, parameters):
    """Define the differential equation for the rotational rate of the planet.

    Args:
        q (list): vector defining op
        t (float): time
        parameters (dict): Dictionary that contains all the parameters for the ODEs.

    Returns:
        list: rotational rate of the planet
    """

    om = q[0]
    # Primary properties
    alpha_planet = parameters['planet_alpha']
    beta_planet = parameters['planet_beta']
    rigidity = parameters['rigidity']
    coeff_planet = parameters['coeff_planet']
    args = parameters['args']

    # Dynamic parameter
    npp = parameters['npp']
    e = parameters['e']
    Mp = parameters['mp']

    # Secondary properties planet
    # Rp = Mp2Rp(Mp, mpo, rpo)
    Rp = Mp2Rp(Mp, t, **args)
    epsilon_planet = om / omegaCritic(Mp, Rp)
    alpha_planet = alpha_planet * parameters['Rp'] / Rp
    beta_planet = beta_planet * parameters['Mp'] / Mp

    if not args['planet_internal_evolution']:
        k2q_planet = args["planet_k2q"]
    else:
        k2q_planet_envelope = k2Q_planet_envelope(alpha_planet, beta_planet, epsilon_planet)
        k2q_planet_core = k2Q_planet_core(rigidity, alpha_planet, beta_planet, Mp, Rp)
        k2q_planet = k2q_planet_envelope + k2q_planet_core

    # eccentricity polinomials
    f3ee = f3e(e)
    f4ee = f4e(e)

    dompdt = ((3.0 * npp**4.0 * k2q_planet * Rp**3.0 / (coeff_planet * GCONST * Mp))
              * (f3ee - (f4ee * om / npp)))

    return [dompdt]


def domsdt(q, t, parameters):
    """Define the differential equation for the rotational rate of the star.

    Args:
        q (list): vector defining os
        t (float): time
        parameters (dict): Dictionary that contains all the parameters for the ODEs.

    Returns:
        list: rotational rate of the star
    """

    os = q[0]

    # Primary properties
    Ms = parameters['Ms']
    Rs = parameters['Rs']
    os_ini = parameters['os_ini']
    star_age = parameters['star_age']
    os_saturation = parameters['os_saturation']
    alpha_star = parameters['star_alpha']
    beta_star_ini = parameters['star_beta']
    coeff_star = parameters['coeff_star']
    sun_mass_loss_rate = parameters['sun_mass_loss_rate']
    sun_omega = parameters['sun_omega']
    args = parameters['args']

    # Dynamic parameter
    npp = parameters['npp']
    e = parameters['e']
    Mp = parameters['mp']

    # Secondary properties star
    epsilon_star = os / omegaCritic(Ms, Rs)

    kappa = kappa_braking(os, star_age)
    osdt_braking = omegadt_braking(kappa, os, os_saturation, os_ini)

    if not args['star_internal_evolution']:
        k2q_star = args['star_k2q']
    else:
        Ms_ini = Ms
        Ms = Ms - mloss_star(Rs, Ms, os, sun_mass_loss_rate, sun_omega) * t
        beta_star = beta_star_ini * Ms_ini / Ms
        k2q_star = k2Q_star_envelope(alpha_star, beta_star, epsilon_star)

    # terms of eccentricity
    f3ee = f3e(e)
    f4ee = f4e(e)

    # Dobbs-Dixon 2004
    domsdt = ((3.0 * npp**4.0 * k2q_star * Rs**3.0 * Mp**2.0
               / (coeff_star * GCONST * Ms * Ms**2.0))
              * (f3ee - f4ee * os / npp)) + osdt_braking

    return [domsdt]


def dmpdt(q, t, parameters):
    """Define the differential equation for the planetary mass loss.

    Args:
        q (list): vector defining mp
        t (float): time
        parameters (dict): Dictionary that contains all the parameters for the ODEs.

    Returns:
        list: mass of the planet
    """

    mp = q[0]

    # Primary properties
    # star
    args = parameters['args']
    Ls = parameters['Ls']
    Ms = parameters['Ms']
    Rs = parameters['Rs']
    sun_mass_loss_rate = parameters['sun_mass_loss_rate']
    sun_omega = parameters['sun_omega']

    # Secondary properties of the planet
    Mp = parameters['mp']
    # Rp = Mp2Rp(mp, mpo, rpo)
    Rp = Mp2Rp(Mp, t, **args)
    npp = parameters['npp']
    app = mean2axis(npp, Ms, Mp)
    os = parameters['os']

    mloss_atmosp = mloss_atmo(t, Ls, app, mp, Rp)
    mloss_drag = mloss_dragging(app, Rp, Rs, Ms, os, sun_mass_loss_rate, sun_omega)

    dmpdt = -mloss_atmosp - mloss_drag
    return [dmpdt]


def solution_star_planet(q, t, parameters):
    """Define the coupled differential equation for the system of EDOs.

    Args:
        q (list): vector defining np
        t (float): time
        parameters (dict): Dictionary that contains all the parameters for the ODEs.

    Returns:
        list: mean motion of the planet
    """
    npp = q[0]
    om = q[1]
    e = q[2]
    os = q[3]
    mp = q[4]

    parameters['npp'] = npp
    parameters['om'] = om
    parameters['e'] = e
    parameters['os'] = os
    parameters['mp'] = mp
    dnpdt_sol = dnpdt([npp], t, parameters)
    dompdt_sol = dompdt([om], t, parameters)
    depdt_sol = depdt([e], t, parameters)
    domsdt_sol = domsdt([os], t, parameters)
    dmpdt_sol = dmpdt([mp], t, parameters)

    return dnpdt_sol + dompdt_sol + depdt_sol + domsdt_sol + dmpdt_sol
