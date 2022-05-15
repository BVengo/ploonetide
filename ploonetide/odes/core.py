#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

from util import *


#############################################################
# CANONICAL UNITS TRANSFORMATION
#############################################################
def canonic_units(**kwargs):
    G = const.G
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
    """Writes scientific notatino formater for plots

    Returns:
        string: Scientific notation of input string
    """
    a, b = f"{x:.1e}".split("e")
    b = int(b)
    return rf'${a} \times 10^{{{b}}}$'


#############################################################
# SPECIFIC TEXOMOONS ROUTINES
#############################################################
def k2Q(alpha, beta, epsilon, **args):
    """
      Source: Mathis, 2015
      alpha = Rc/Rp
      beta = Mc/Mp
      epsilon = Omega/Omega_crit
      args = contains behavior
    """
    # if args["qk2q"]==0:return args["k2q"]

    fac0 = alpha**3
    fac1 = alpha**5
    fac2 = fac1 / (1 - fac1)
    if beta > 0:
        gamma = fac0 * (1 - beta) / (beta * (1 - fac0))
        fac3 = (1 - gamma) / gamma * fac0
    else:
        fac3 = -1 * fac0

    k2q = 100 * np.pi / 63 * epsilon**2 * fac2 * (1 + fac3) / (1 + 5. / 2 * fac3)**2

    return k2q


def Mp2Rp(Mp, t, **args):
    """
    args = contains behavior
    """
    if args["qRp"] == 0:
        return args["Rp"]
    if Mp >= PLANETS.Jupiter.M:
        rad = PLANETS.Jupiter.R
    else:
        rad = PLANETS.Saturn.R
    Rp = rad * A * ((t / YEAR + t0) / C)**B
    return Rp


def aRoche(Mp, densPart=3000, rfac=2.0, **args):
    # Since Roche radius does not depend on R this is a hypotetical one
    Rp = PLANETS.Saturn.R
    # Planet average density
    densP = Mp / ((4. / 3) * np.pi * Rp**3)
    # Roche radius
    ar = rfac * Rp * (densPart / densP)**(-1.0 / 3.0)
    return ar


def aRoche_solid(Mp, Mm, Rm):

    return Rm * (2. * Mp / Mm)**(1. / 3.)


def alpha2beta(Mp, alpha, **args):
    beta = KP * (Mp / PLANETS.Saturn.M)**DP * alpha**BP
    return beta


def omegaAngular(P):
    return 2 * np.pi / P


def omegaCritic(M, R):
    Oc = np.sqrt(GCONST * M / R**3)
    return Oc


def equil_temp(Ts, Rs, a, Ab):
    T_eq = Ts * (Rs / (2 * a))**0.5 * (1 - Ab)**0.25
    return T_eq


def semiMajorAxis(P, M, m):
    a = (GCONST * (M + m) * P**2.0 / (2.0 * np.pi)**2.0)**(1.0 / 3.0)
    return a


def meanMotion(a, M, m):
    n = (GCONST * (M + m) / a**3.0)**0.5
    return n


def mean2axis(N, M, m):
    return (GCONST * (M + m) / N**2.0)**(1.0 / 3.0)


def gravity(M, R):

    return GCONST * M / R**2.


def density(M, R):

    return M / (4. / 3 * np.pi * R**3.)


def surf_temp(dEdt, Rm, sigmasb=const.sigma):

    return (dEdt / (4. * np.pi * Rm**2. * sigmasb))**0.25


def stellar_lifespan(Ms):
    return 10 * (MSUN / Ms)**2.5 * GYEAR


def im_k2(T, omeg=None, densm=None, Mm=None, Rm=None, E_act=300e3, melt_fr=0.5, B=25,
          Ts=1600, Tb=1800, T1=2000, Rg=const.gas_constant, shear_Conv=1., visc_Conv=1.,
          eta_o=1.6e5):

    if T < Ts:
        mu = mu_below_Ts(shear_Conv=shear_Conv)
        eta = eta_below_Ts(T, E_act=E_act, Rg=Rg, eta_o=eta_o, visc_Conv=visc_Conv)

    elif Ts <= T < Tb:
        mu = mu_between_Ts_Tb(T, shear_Conv=shear_Conv)
        eta = eta_between_Ts_Tb(T, E_act=E_act, Rg=Rg, melt_fr=melt_fr, B=B, eta_o=eta_o,
                                visc_Conv=visc_Conv)

    elif Tb <= T < T1:
        mu = mu_between_Tb_T1(shear_Conv=shear_Conv)
        eta = eta_between_Tb_T1(T, melt_fr=melt_fr, visc_Conv=visc_Conv)

    else:
        mu = mu_above_T1(shear_Conv=shear_Conv)
        eta = eta_above_T1(T, visc_Conv=visc_Conv)

    numerator = 57 * eta * omeg

    deno_brackets = 1. + (1. + 19. * mu
                          / (2. * densm * gravity(Mm, Rm) * Rm))**2. * (eta * omeg / mu)**2.

    denominator = 4 * densm * gravity(Mm, Rm) * Rm * deno_brackets

    return -numerator / denominator


def mu_below_Ts(shear_Conv=1.):

    return 50 * const.giga * shear_Conv


def eta_below_Ts(T, E_act=300e3, eta_o=1.6e5, Rg=const.gas_constant, visc_Conv=1.):

    return eta_o * np.exp(E_act / (Rg * T)) * visc_Conv


def mu_between_Ts_Tb(T, mu1=8.2e4, mu2=-40.6, shear_Conv=1.):

    return 10**(mu1 / T + mu2) * shear_Conv


def eta_between_Ts_Tb(T, E_act=300e3, melt_fr=0.5, B=25, eta_o=1.6e5,
                      Rg=const.gas_constant, visc_Conv=1.):

    return eta_o * np.exp(E_act / (Rg * T)) * np.exp(-B * melt_fr) * visc_Conv


def mu_between_Tb_T1(shear_Conv=1.):

    return 1e-7 * shear_Conv


def eta_between_Tb_T1(T, melt_fr=0.5, visc_Conv=1.):

    return 1e-7 * np.exp(40000. / T) * (1.35 * melt_fr - 0.35)**(-5. / 2.) * visc_Conv


def mu_above_T1(shear_Conv=1.):

    return 1e-7 * shear_Conv


def eta_above_T1(T, visc_Conv=1.):

    return 1e-7 * np.exp(40000. / T) * visc_Conv


def e_tidal(T, nm, omeg=None, densm=None, Mm=None, Rm=None, E_act=300e3, Rg=const.gas_constant,
            melt_fr=0.5, B=25, Ts=1600, Tb=1800, T1=2000, Mp=None, eccm=None, eta_o=1.6e5,
            shear_Conv=1., visc_Conv=1.):

    term_1 = -10.5 * im_k2(T, omeg=nm, densm=densm, Mm=Mm, Rm=Rm, E_act=E_act, Rg=Rg,
                           melt_fr=melt_fr, B=B, Ts=Ts, Tb=Tb, T1=T1, eta_o=eta_o,
                           shear_Conv=shear_Conv, visc_Conv=visc_Conv)

    term_2 = Rm**5. * nm**5. * eccm**2. / GCONST

    dedt = term_1 * term_2

    return dedt


#############################################################
# DIFFERENTIAL EQUATIONS
#############################################################
def dnmdt(q, t, parameters):

    nm = q[0]

    # Evolving conditions
    args = parameters["args"]    

    # Primary properties
    Mp = parameters["Mp"]
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    Mm = parameters["Mm"]

    # Dynamic parameter
    op = parameters["op"]
    if parameters["em_ini"] == 0.0:
        eccm = 0.0
    else:
        eccm = parameters["eccm"]

    # Secondary properties
    if args["qRp"] == 0:
        Rp = args["Rp"]

    else:
        Rp = Mp2Rp(Mp, t, **args)
        alpha = alpha * args["Rp"] / Rp

    epsilon = op / omegaCritic(Mp, Rp)
    # beta=alpha2beta(Mp,alpha,**args)
    if args["qk2q"] == 0:
        k2q = args["k2q"]
    else:
        k2q = k2Q(alpha, beta, epsilon, **args)

    if parameters["em_ini"] == 0.0:
        dnmdt = (-9. / 2 * k2q * Mm * Rp**5 / (GCONST**(5. / 3) * Mp**(8. / 3))
                 * nm**(16. / 3) * np.sign(op - nm))   
    else:
        dnmdt = 9. * nm**(16. / 3.) * k2q * Mm * Rp**5. /\
        (Mp * (GCONST * (Mp + Mm))**(5. / 3.)) *\
        ((1. + 23. * eccm**2.) - (1. + 13.5 * eccm**2.) * op / nm)
    

    return [dnmdt]


def demdt(q, t, parameters):

    eccm = q[0]

    # Evolving conditions
    args = parameters["args"]    

    # Primary properties
    Mp = parameters["Mp"]
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    Mm = parameters["Mm"]

    # Dynamic parameter
    op = parameters["op"]
    nm = parameters["nm"]

    # Secondary properties
    if args["qRp"] == 0:
        Rp = args["Rp"]

    else:
        Rp = Mp2Rp(Mp, t, **args)
        alpha = alpha * args["Rp"] / Rp

    epsilon = op / omegaCritic(Mp, Rp)
    # beta=alpha2beta(Mp,alpha,**args)
    if args["qk2q"] == 0:
        k2q = args["k2q"]
    else:
        k2q = k2Q(alpha, beta, epsilon, **args)

    demdt = -27. * nm**(13. / 3.) * eccm * k2q * Mm * Rp**5. / (Mp * (GCONST * (Mp + Mm))**(5. / 3.)) *\
        (1. - 11. / 18. * op / nm)

    return [demdt]


def dopdt(q, t, parameters):

    op = q[0]

    # Evolving conditions
    args = parameters["args"]    

    # Primary properties
    Mp = parameters["Mp"]
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    Mm = parameters["Mm"]
    npp = parameters["npp"]

    # Dynamic parameter
    nm = parameters["nm"]
    npp = parameters["npp"]

    # Secondary properties
    if args["qRp"] == 0:
        Rp = args["Rp"]

    else:
        Rp = Mp2Rp(Mp, t, **args)
        alpha = alpha * args["Rp"] / Rp

    epsilon = op / omegaCritic(Mp, Rp)
    # beta=alpha2beta(Mp,alpha,**args)
    if args["qk2q"] == 0:
        k2q = args["k2q"]
    else:
        k2q = k2Q(alpha, beta, epsilon, **args)

    dopdt = -3. / 2. * k2q * Rp**3 / (GR * GCONST) *\
        (Mm**2. * nm**4. * np.sign(op - nm) / Mp**3
         + npp**4. * np.sign(op - npp) / Mp)

    # dopdt = -3. / 2. * k2q * Rp**3 / (GR * GCONST) *\
    #     (Mm**2. * nm**4. * np.sign(op - nm) / Mp**3
    #      + (GCONST * Ms)**2. * np.sign(op - nmp) / (Mp * ap**6.))

    return [dopdt]


def dnpdt(q, t, parameters):

    npp = q[0]

    # Evolving conditions
    args = parameters["args"]    

    # Primary properties
    Ms = parameters["Ms"]
    Mp = parameters["Mp"]
    alpha = parameters["alpha"]
    beta = parameters["beta"]

    # Dynamic parameter
    op = parameters["op"]

    # Secondary properties
    if args["qRp"] == 0:
        Rp = args["Rp"]

    else:
        Rp = Mp2Rp(Mp, t, **args)
        alpha = alpha * args["Rp"] / Rp

    epsilon = op / omegaCritic(Mp, Rp)
    # beta=alpha2beta(Mp,alpha,**args)
    if args["qk2q"] == 0:
        k2q = args["k2q"]
    else:
        k2q = k2Q(alpha, beta, epsilon, **args)

    dnpdt = (-9. / 2 * k2q * Rp**5 / (GCONST**(5. / 3.) * Mp * Ms**(2. / 3.))
             * npp**(16. / 3) * np.sign(op - npp))

    return [dnpdt]


#############################################################
# INTEGRATION OF THE TIDAL HEAT
#############################################################
def dEmdt(q, t, parameters):

    E = q[0]

    # General parameters
    E_act = parameters["E_act"]
    Rg = parameters["Rg"]
    shear_Conv = parameters["shear_Conv"]
    visc_Conv = parameters["visc_Conv"]
    B = parameters["B"]
    Ts = parameters["Ts"]
    Tb = parameters["Tb"]
    T1 = parameters["T1"]

    # Moon parameters
    densm = parameters["densm"]
    Mm = parameters["Mm"]
    Rm = parameters["Rm"]
    melt_fr = parameters["melt_fr"]

    # Dynamic parameters
    Tm = parameters["Tm"]
    nm = parameters["nm"]

    if parameters["em_ini"] == 0.0:
        eccm = 0.0
    else:
        eccm = parameters["eccm"]

    dEdt = e_tidal(Tm, nm, densm=densm, Mm=Mm, Rm=Rm, E_act=E_act, Rg=Rg, melt_fr=melt_fr, B=B,
                   Ts=Ts, Tb=Tb, T1=T1, eccm=eccm, shear_Conv=shear_Conv, visc_Conv=visc_Conv)

    return [dEdt]


#############################################################
# INTEGRATION OF THE TEMPERATURE
#############################################################
def dTmdt(q, t, parameters):

    Tm = q[0]

    # General parameters
    E_act = parameters["E_act"]
    Rg = parameters["Rg"]
    sigmasb = parameters["sigmasb"]
    shear_Conv = parameters["shear_Conv"]
    visc_Conv = parameters["visc_Conv"]
    B = parameters["B"]
    Ts = parameters["Ts"]
    Tb = parameters["Tb"]
    T1 = parameters["T1"]
    Cp = parameters["Cp"]
    ktherm = parameters["ktherm"]
    Rac = parameters["Rac"]
    a2 = parameters["a2"]
    alpha_exp = parameters["alpha_exp"]

    # Moon parameters
    densm = parameters["densm"]
    Mm = parameters["Mm"]
    Rm = parameters["Rm"]
    melt_fr = parameters["melt_fr"]

    # Dynamic parameter
    nm = parameters["nm"]

    if parameters["em_ini"] == 0.0:
        eccm = 0.0
    else:
        eccm = parameters["eccm"]

    dEdt = e_tidal(Tm, nm, densm=densm, Mm=Mm, Rm=Rm, E_act=E_act, Rg=Rg, melt_fr=melt_fr,
                   B=B, Ts=Ts, Tb=Tb, T1=T1, eccm=eccm, shear_Conv=shear_Conv, visc_Conv=visc_Conv)

    if Tm < Ts:
        eta = eta_below_Ts(Tm, E_act=E_act, Rg=Rg, visc_Conv=visc_Conv)

    elif Ts <= Tm < Tb:
        eta = eta_between_Ts_Tb(Tm, E_act=E_act, Rg=Rg, melt_fr=melt_fr, B=B, visc_Conv=visc_Conv)

    elif Tb <= Tm < T1:
        eta = eta_between_Tb_T1(Tm, melt_fr=melt_fr, visc_Conv=visc_Conv)

    else:
        eta = eta_above_T1(Tm, visc_Conv=visc_Conv)

    # Calculation of convection
    kappa = ktherm / (densm * Cp)

    C = Rac**0.25 / (2 * a2) * (alpha_exp * gravity(Mm, Rm) * densm
                                / (eta * kappa * ktherm))**-0.25
    qBL = (ktherm * (Tm - surf_temp(dEdt, Rm, sigmasb=sigmasb)) / C)**(4. / 3.)

    # qBL = ktherm / 2. * (densm * gravity(Mm, Rm) * alpha_exp / (kappa * eta))**(1. / 3.) *\
    #     (E_act / (Rg * Tm**2.))**(-4. / 3.)

    coeff = 4. / 3. * np.pi * (Rm**3. - (0.4 * Rm)**3.) * densm * Cp

    dTdt = (-qBL + dEdt) / coeff

    return [dTdt]


#############################################################
# INTEGRATION OF THE WHOLE SYSTEM
#############################################################
def global_differential_equation(q, t, parameters):

    args = parameters["args"]

    nm = q[0]
    op = q[1]
    npp = q[2]
    Tm = q[3]
    Em = q[4]

    if parameters["em_ini"] != 0.0:
        eccm = q[5]
        parameters["eccm"] = eccm

    parameters["nm"] = nm
    parameters["op"] = op
    parameters["npp"] = npp
    parameters["Tm"] = Tm
    parameters["Em"] = Em        

    dnmdtm = dnmdt([nm], t, parameters)
    dopdtp = dopdt([op], t, parameters)
    dnpdtp = dnpdt([npp], t, parameters)
    dTmdtm = dTmdt([Tm], t, parameters)
    dEmdtm = dEmdt([Em], t, parameters)

    solution = dnmdtm + dopdtp + dnpdtp + dTmdtm + dEmdtm
    
    if parameters["em_ini"] != 0.0:
        demdtm = demdt([eccm], t, parameters)
        solution = dnmdtm + dopdtp + dnpdtp + dTmdtm + dEmdtm + demdtm

    return solution
