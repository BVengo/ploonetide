import numpy as np

from collections import namedtuple

from ploonetide.utils.constants import *


#############################################################
# SPECIFIC ROUTINES
#############################################################
def k2Q_star_envelope(alpha, beta, epsilon):
    """Calculate tidal heat function for a stellar envelope (Source: Mathis, 2015).

      Args:
          alpha (float): star's core size fraction [Rc/Rs]
          beta (float): star's core mass fraction [Mc/Ms]
          epsilon (float): star's rotational rate [Omega/Omega_crit]
          args (list, optional): contains behaviour

      Returns:
          float: tidal heat function
    """
    gamma = alpha**3. * (1 - beta) / (beta * (1 - alpha**3.))

    line1 = 100 * np.pi / 63 * epsilon**2 * (alpha**5. / (1 - alpha**5.)) * (1 - gamma)**2.
    line2 = ((1 - alpha)**4.0 * (1 + 2 * alpha + 3 * alpha**2. + 1.5 * alpha**3.)**2.0
             * (1 + (1 - gamma) / gamma * alpha**3.))
    line3 = (1 + 1.5 * gamma + 2.5 / gamma * (1 + 0.5 * gamma - 1.5 * gamma**2.)
             * alpha**3. - 9. / 4. * (1 - gamma) * alpha**5.)

    k2q1 = line1 * line2 / line3**2.0

    return k2q1


def k2Q_planet_envelope(alpha, beta, epsilon):
    """Calculate tidal heat function for the planet's envelope (Source: Mathis, 2015).

      Args:
          alpha (float): planet's core size fraction [Rc/Rp]
          beta (float): planet's core mass fraction [Mc/Mp]
          epsilon: planetary rotational rate (Omega/Omega_crit)

      Returns:
          float: tidal heat function

    """
    fac0 = alpha**3.0
    fac1 = alpha**5.0
    fac2 = fac1 / (1 - fac1)

    gamma = fac0 * (1 - beta) / (beta * (1 - fac0))
    fac3 = (1 - gamma) / gamma * fac0

    k2q = 100 * np.pi / 63 * epsilon**2 * fac2 * (1 + fac3) / (1 + 5. / 2 * fac3)**2

    return k2q


def k2Q_planet_core(G, alpha, beta, Mp, Rp):
    """Calculate tidal heat function for the planete's core (Source: Mathis, 2015).

    Args:
        G (float): planet's core rigidity
        alpha (float): planet's core size fraction [Rc/Rp]
        beta (float): planet's core mass fraction [Mc/Mp]
        Mp (float): planet's mass [SI units]
        Rp (float): planet's radius [SI units]

    Returns:
        float: tidal heat function
    """
    gamma = alpha**3.0 * (1 - beta) / (beta * (1 - alpha**3.0))

    AA = 1.0 + 2.5 * gamma**(-1.0) * alpha**3.0 * (1.0 - gamma)
    BB = alpha**(-5.0) * (1.0 - gamma)**(-2.0)
    CC = (38.0 * np.pi * (alpha * Rp)**4.0) / (3.0 * GCONST * (beta * Mp)**2.0)
    DD = (2.0 / 3.0) * AA * BB * (1.0 - gamma) * (1.0 + 1.5 * gamma) - 1.5

    num = np.pi * G * (3.0 + 2.0 * AA)**2.0 * BB * CC
    den = DD * (6.0 * DD + 4.0 * AA * BB * CC * G)
    k2qcore = num / den
    return k2qcore


# ############RODRIGUEZ 2011########################
def S(kQ1, Mp, Ms, Rs):
    return (9 * kQ1 * Mp * Rs**5.0) / (Ms * 4.0)


def p(kQ, Mp, Ms, Rp):
    return (9 * kQ * Ms * Rp**5.0) / (Mp * 2.0)


def D(pp, SS):
    return pp / (2 * SS)
# ############RODRIGUEZ 2011########################


def Mp2Rp(Mp, t):
    if Mp >= PLANETS.Jupiter.M:
        rad = PLANETS.Jupiter.R
    else:
        rad = PLANETS.Saturn.R
    Rp = rad * A * ((t / YEAR + t0) / C)**B
    return Rp


def mloss_atmo(t, Ls, a, Mp, Rp):
    """Calculate loss of mass in the atmoshpere of the planet.

    Args:
        t (float): time
        Ls (float): stellar luminosity [W]
        a (float): planetary semi-major axis [m]
        Mp (float): mass of the planet [kg]
        Rp (float): radius of the planet [m]

    Returns:
        float: loss rate of atmospheric mass
    """
    #  Zuluaga et. al (2012)
    ti = 0.06 * GYEAR * (Ls / LSUN)**-0.65

    if t < ti:
        Lx = 6.3E-4 * Ls
    else:
        Lx = 1.8928E28 * t**(-1.55)
    # Sanz-forcada et. al (2011)
    Leuv = 10**(4.8 + 0.86 * np.log10(Lx))
    k_param = 1.0  # Sanz-forcada et. al (2011)

    lxuv = (Lx + Leuv) * 1E-7
    fxuv = lxuv / (4 * np.pi * a**2.0)

    num = np.pi * Rp**3.0 * fxuv
    deno = GCONST * Mp * k_param
    return num / deno


def mloss_dragging(a, Rp, Rs, Ms, oms, sun_mass_loss_rate, sun_omega):
    """Calculate mass loss in the planet fue to atmospheric dragging."""
    alpha_eff = 0.3  # Zendejas et. al (2010) Venus

    return (Rp / a)**2.0 * mloss_star(Rs, Ms, oms, sun_mass_loss_rate, sun_omega) * alpha_eff / 2.0


def mloss_star(Rs, Ms, oms, sun_mass_loss_rate, sun_omega):
    """Calculate the loss of mass in the star due to wind."""
    # smlr_sun = 1.4E-14 * MSUN / YEAR  # Zendejas et. al (2010) - smlr sun
    # oms_sun = 2.67E-6
    m_loss = (sun_mass_loss_rate * (Rs / RSUN)**2.0
              * (oms / sun_omega)**1.33 * (Ms / MSUN)**-3.36)

    return m_loss


def omegadt_braking(kappa, OS, OS_saturation, osini, dobbs=False):
    """Calculate the rate of magnetic braking in th star."""
    if dobbs:
        gam = 1.0
        tao = GYEAR
        odt_braking = -gam / 2 * (osini / tao) * (OS / osini)**3.0
        return odt_braking

    if isinstance(OS, np.ndarray):
        odt_braking = []
        for k, o in zip(kappa, OS):
            odtb = []
            for i in range(len(k)):
                odtb.append(-k[i] * o[i] * min(o[i], OS_saturation)**2.0)
            odt_braking.append(np.array(odtb))
        return odt_braking
    odt_braking = -kappa * OS * min(OS, OS_saturation)**2.0

    return odt_braking


def kappa_braking(OS, stellar_age, skumanich=True, alpha=0.495):
    """Calulate the kappa coefficient for mangnetic braking."""
    alpha_s = 0.5  # Skumanich (1972)
    kappa = OS**-2.0 / (2.0 * stellar_age)  # Weber-Davis

    if not skumanich:
        alpha_s = alpha  # Brown et. al (2011)
        kappa = OS**(-1.0 / alpha_s) / (stellar_age / alpha_s)  # Brown (2011)
        return kappa
    return kappa


def aRoche(Mp, densPart=3000, rfac=2.0, **args):
    """Calculate the Roche radius in term of the densities."""
    Rp = PLANETS.Saturn.R  # Since Roche radius does not depend on R this is a hypotetical one
    # Planet average density
    densP = Mp / ((4. / 3) * np.pi * Rp**3)
    # Roche radius
    ar = rfac * Rp * (densPart / densP)**(-1.0 / 3.0)
    return ar


def aRoche_solid(Mp, Mm, Rm):
    """Calculate the Roche radius using the masses.

    Args:
        Mp (float): Planet's mass [kg]
        Mm (float): Moon mass [kg]
        Rm (float): Moon radius [kg]

    Returns:
        float: Roche radius of the body with Mm.
    """
    return Rm * (2. * Mp / Mm)**(1. / 3.)


def hill_radius(a, Mp, Ms):
    return a * (Mp / (3.0 * Ms))**(1.0 / 3.0)


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


def luminosity(R, T):
    L = 4 * np.pi * R**2.0 * stefan_b_constant * T**4.0
    return L


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


def surf_temp(dEdt, Rm):

    return (dEdt / (4. * np.pi * Rm**2. * stefan_b_constant))**0.25


def stellar_lifespan(Ms):
    """Calculate lifespan of a star.

    Args:
        Ms (float): Stellar mass [kg]

    Returns:
        float: lifespan of the star [s]
    """
    return 10 * (MSUN / Ms)**2.5 * GYEAR


# ###################DOBS-DIXON 2004#######################
def f1e(ee):
    numer = (1 + 3.75 * ee**2.0 + 1.875 * ee**4.0 + 0.078125 * ee**6.0)
    deno = (1 - ee**2.0)**6.5
    return numer / deno


def f2e(ee):
    numer = (1 + 1.5 * ee**2.0 + 0.125 * ee**4.0)
    deno = (1 - ee**2.0)**5.0
    return numer / deno


def f3e(ee):
    numer = (1 + 7.5 * ee**2.0 + 5.625 * ee**4.0 + 0.3125 * ee**6.0)
    deno = (1 - ee**2.0)**6.0
    return numer / deno


def f4e(ee):
    numer = (1 + 3 * ee**2.0 + 0.375 * ee**4.0)
    deno = (1 - ee**2.0)**4.5
    return numer / deno


def factorbet(ee, OM, OS, N, KQ, KQ1, MP, MS, RP, RS):
    fac1 = f1e(ee) - 0.611 * f2e(ee) * (OM / N)
    fac2 = f1e(ee) - 0.611 * f2e(ee) * (OS / N)
    lamb = (KQ / KQ1) * (MS / MP)**2.0 * (RP / RS)**5.0
    return 18.0 / 7.0 * (fac1 + fac2 / lamb)


def power(ee, aa, KQ, Ms, Rp):
    keys = (GCONST * Ms)**1.5 * ((2 * Ms * Rp**5.0 * ee**2.0 * KQ) / 3)
    coeff = 15.75 * aa**(-7.5)
    return coeff * keys
# ###################DOBS-DIXON 2004#######################


def find_moon_fate(t, am, amr, porb, Mp, Ms):
    try:
        pos = np.where(am <= amr)[0][0]
        rt_time = t[pos] / GYEAR
        print(f'Moon crosses Roche radius in {rt_time:.2f} Gyr')
    except IndexError:
        try:
            ap = semiMajorAxis(porb, Ms, Mp)
            r_hill = hill_radius(ap, Mp, Ms)
            pos = np.where(am >= 0.48 * r_hill)[0][0]
            rt_time = t[pos] / GYEAR
            print(f'Moon escapes in {rt_time:.2f} Gyr')
        except IndexError:
            pos = -1
            rt_time = np.max(t)
            print('Moon migrates too slow and never escapes or crosses Roche radius.')

    Outputs = namedtuple('Outputs', 't index')

    return Outputs(rt_time, pos)

def im_k2(T, omeg, densm, Mm, Rm, E_act, melt_fr, B, Ts, Tb, Tl):

    if T < Ts:
        mu = mu_below_Ts()
        eta = eta_below_Ts(T, E_act=E_act)

    elif Ts <= T < Tb:
        mu = mu_between_Ts_Tb(T)
        eta = eta_between_Ts_Tb(T, E_act=E_act, melt_fr=melt_fr, B=B)

    elif Tb <= T < Tl:
        mu = mu_between_Tb_Tl()
        eta = eta_between_Tb_Tl(T, melt_fr=melt_fr)

    else:
        mu = mu_above_Tl()
        eta = eta_above_Tl(T)

    numerator = 57 * eta * omeg

    deno_brackets = 1. + (1. + 19. * mu
                          / (2. * densm * gravity(Mm, Rm) * Rm))**2. * (eta * omeg / mu)**2.

    denominator = 4 * densm * gravity(Mm, Rm) * Rm * deno_brackets

    return -numerator / denominator


def mu_below_Ts():

    return 50 * GYEAR


def eta_below_Ts(T, E_act, eta_o=1.6E5):

    return eta_o * np.exp(E_act / (gas_constant * T))


def mu_between_Ts_Tb(T, mu1=8.2e4, mu2=-40.6):

    return 10**(mu1 / T + mu2)


def eta_between_Ts_Tb(T, E_act, melt_fr, B, eta_o=1.6E5):

    return eta_o * np.exp(E_act / (gas_constant * T)) * np.exp(-B * melt_fr)


def mu_between_Tb_Tl():

    return 1e-7


def eta_between_Tb_Tl(T, melt_fr):

    return 1e-7 * np.exp(40000. / T) * (1.35 * melt_fr - 0.35)**(-5. / 2.)


def mu_above_Tl():

    return 1e-7


def eta_above_Tl(T):

    return 1e-7 * np.exp(40000. / T)


def e_tidal(T, nm, omeg=None, densm=None, Mm=None, Rm=None, E_act=300E3,
            melt_fr=0.5, B=25, Ts=1600, Tb=1800, Tl=2000, Mp=None, eccm=None):

    term_1 = -10.5 * im_k2(T, omeg=nm, densm=densm, Mm=Mm, Rm=Rm, E_act=E_act, melt_fr=melt_fr,
                           B=B, Ts=Ts, Tb=Tb, Tl=Tl)

    term_2 = Rm**5. * nm**5. * eccm**2. / GCONST

    dedt = term_1 * term_2

    return dedt
