import numpy as np
import astropy.units as u

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


def hill_radius(a, e, m, M):
    return a * (1 - e) * (m / (3.0 * M))**(1.0 / 3.0)


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
    return u.Quantity(L, u.W)


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


def surf_temp(flux):

    return (flux / stefan_b_constant)**0.25


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


def find_moon_fate(t, am, am_roche, ap_hill):
    try:
        pos = np.where(am <= am_roche)[0][0]
        rt_time = t[pos] / MYEAR
        label = 'crosses'
        print(f'Moon {label} the Roche limit in {rt_time:.6f} Myr')
    except IndexError:
        try:
            pos = np.where(am >= ap_hill)[0][0]
            rt_time = t[pos] / MYEAR
            label = 'escapes'
            print(f'Moon {label} from the planetary Hill radius in {rt_time:.6f} Myr')
        except IndexError:
            pos = -1
            rt_time = np.max(t) / MYEAR
            label = "stalls"
            print('Moon migrates too slow and never escapes the Hill radius or crosses the Roche limit.')

    Outputs = namedtuple('Outputs', 'time index label')

    return Outputs(rt_time, pos, label)


# def cross_roche_limit(t, y):
#     return meanMotion()


def mu_below_T_solidus():
    # Shear modulus [Pa]
    return 50 * const.giga


def eta_o(E_act):
    # Viscosity [Pa s]
    # defining viscosity for Earth at T0 = 1000K [Pa*s] (Henning et al 2009)
    eta_set = 1e22
    # eta_set = 1e19 # defining viscosity for Mars at T0 = 1600K [Pa*s] (Shoji & Kurita 2014)
    T0 = 1000.  # defining temperature
    return eta_set / np.exp(E_act / (gas_constant * T0))

def eta_below_T_solidus(T, E_act):

    return eta_o(E_act=E_act) * np.exp(E_act / (gas_constant * T))


def mu_between_T_solidus_T_breakdown(T, mu1=8.2E4, mu2=-40.6):
    # Fischer & Spohn (1990), Eq. 16
    return 10**(mu1 / T + mu2)


def eta_between_T_solidus_T_breakdown(T, E_act, melt_fr, B):
    # Moore (2003)
    return eta_o(E_act=E_act) * np.exp(E_act / (gas_constant * T)) * np.exp(-B * melt_fr)


def mu_between_T_breakdown_T_liquidus():
    # Moore (2003)
    return 1E-7


def eta_between_T_breakdown_T_liquidus(T, melt_fr):
    # Moore (2003)
    return 1E-7 * np.exp(40000. / T) * (1.35 * melt_fr - 0.35)**(-5. / 2.)


def mu_above_T_liquidus():
    # Moore (2003)
    return 1E-7


def eta_above_T_liquidus(T):
    # Moore (2003)
    return 1E-7 * np.exp(40000. / T)


def tidal_heat(T, nm, eccm, parameters):

    # General parameters
    E_act = parameters['E_act']
    B = parameters['B']
    T_solidus = parameters['T_solidus']
    T_breakdown = parameters['T_breakdown']
    T_liquidus = parameters['T_liquidus']

    # Moon properties
    Rm = parameters['Rm']  # Moon radius [m]
    rigidm = parameters['rigidm']  # Effective rigidity of the moon [m^-1 s^-2]

    # Orbital angular frequency of the moon [1/s]
    freq = nm

    if T > T_solidus:
        # melt_fraction: Fraction of melt for ice [No unit]
        melt_fr = (T - T_solidus) / (T_liquidus - T_solidus)  # melt fraction

    if T <= T_solidus:
        mu = mu_below_T_solidus()
        eta = eta_below_T_solidus(T, E_act=E_act)

    elif T_solidus < T <= T_breakdown:
        mu = mu_between_T_solidus_T_breakdown(T)
        eta = eta_between_T_solidus_T_breakdown(T, E_act=E_act, melt_fr=melt_fr, B=B)

    elif T_breakdown < T <= T_liquidus:
        mu = mu_between_T_breakdown_T_liquidus()
        eta = eta_between_T_breakdown_T_liquidus(T, melt_fr=melt_fr)

    else:
        mu = mu_above_T_liquidus()
        eta = eta_above_T_liquidus(T)

    # Imaginary part of the second order Love number, Maxwell model (Henning et al. 2009, table 1)
    if mu == 0:
        k2_Im = 0.
    else:
        numerator = -57 * eta * freq

        denominator = 4 * rigidm * (1. + (1. + 19. * mu / (2. * rigidm))**2. * (eta * freq / mu)**2.)

        k2_Im = numerator / denominator

    # tidal surface flux of the moon [W/m^2] (Fischer & Spohn 1990)
    h_m = (-21. / 2. * k2_Im * Rm**5. * nm**5. * eccm**2. / GCONST) / (4. * np.pi * Rm**2.)

    return (h_m, eta)


def convection_heat(T, eta, parameters):

    # General parameters
    Cp = parameters['Cp']
    ktherm = parameters['ktherm']
    Rac = parameters['Rac']
    a2 = parameters['a2']
    alpha_exp = parameters['alpha_exp']
    d_mantle = parameters['d_mantle']
    T_surface = parameters['T_surface']

    # Moon properties
    rho_m = parameters['densm']  # Density of the moon [kg m^-3]
    g_m = parameters['gravm']  # Gravity of the moon [m s^-2]

    error = False

    delta = 30000.  # Initial guess for boudary layer thickness [m]
    kappa = ktherm / (rho_m * Cp)  # Thermal diffusivity [m^2/s]

    if T == 288.:
        print("___error1___: T = T_surf --> q_BL = 0")
        q_BL = 0
        error = True

    if error:
        return q_BL

    q_BL = ktherm * ((T - T_surface) / delta)  # convective heat flux [W/m^2]

    prev = q_BL + 1.

    difference = abs(q_BL - prev)

    # Iteration for calculating q_BL:
    while difference > 10e-10:
        prev = q_BL

        Ra = alpha_exp * g_m * rho_m * d_mantle**4 * q_BL / (eta * kappa * ktherm)  # Rayleigh number

        # Thickness of the conducting boundary layer [m]
        delta = d_mantle / (2. * a2) * (Ra / Rac)**(-1. / 4.)

        q_BL = ktherm * ((T - T_surface) / delta)  # convective heat flux [W/m^2]

        difference = abs(q_BL - prev)

    return q_BL


def check(difference, prev):

    if difference < 0 and prev > 0:
        T_stable = True
    elif difference == 0 and prev > 0:
        T_stable = True
    elif difference > 0 and prev < 0:
        T_stable = False
    elif difference == 0 and prev < 0:
        T_stable = False
    else:
        T_stable = -1

    return T_stable


def intersection(a, b, nm, eccm, parameters):

    done = False
    again = False
    error = False

    T_equilibrium = 0
    i = 0
    unstable = 0
    c = (a + b) / 2.

    while abs(a - c) >= 0.1:  # 0.1K error allowed

        i = i + 1

        f1a, eta = tidal_heat(a, nm, eccm, parameters)
        f2a = convection_heat(a, eta, parameters)

        f1c, eta = tidal_heat(c, nm, eccm, parameters)
        f2c = convection_heat(c, eta, parameters)

        if f2a == 0 or f2c == 0:
            error = True
            T_equilibrium = -1
            return T_equilibrium

        fa = f1a - f2a

        fc = f1c - f2c

        if (fa * fc) < 0:
            b = c
            c = (a + b) / 2.

        elif (fa * fc) > 0:
            a = c
            c = (a + b) / 2.

        elif (fa * fc) == 0:
            if fa == 0:
                f1a, eta = tidal_heat((a - 0.1), nm, eccm, parameters)
                f2a = convection_heat((a - 0.1), eta, parameters)
                stab = check((f1c - f2c), (f1a - f2a))
                if stab:
                    T_equilibrium = a
                    done = True
                elif not stab:
                    a = a + 0.01
                    unstable = 1
                    again = True
                else:
                    T_equilibrium = -3
                    error = True
            if fc == 0:
                f1c, eta = tidal_heat((c + 0.1), nm, eccm, parameters)
                f2c = convection_heat((c + 0.1), eta, parameters)
                stab = check((f1c - f2c), (f1a - f2a))
                if stab:
                    T_equilibrium = c
                    done = True
                elif not stab:
                    a = c + 0.01
                    unstable = 1
                    again = True
                else:
                    T_equilibrium = -3
                    error = True

        if done:
            break
        if again:
            break
        if error:
            break

    if error:
        if T_equilibrium == -3:
            print("___error3___: T_equilibrium not found??")

    return (T_equilibrium, a, b, again, unstable)


def bisection(nm, eccm, parameters):

    # General parameters
    T_breakdown = parameters['T_breakdown']
    T_liquidus = parameters['T_liquidus']

    a0 = 600.                  # ################ ASK ABOUT THIS OBSCURE PARAMETER #############
    b0 = T_liquidus
    a = a0
    b = T_breakdown
    error = False
    # done = False
    again = False
    T_equilibrium = 0
    i = 0

    c = (a + b) / 2.

    # Find the peak of h_m (derivative=0):
    while abs(a - c) >= 0.01:  # 0.01K error allowed

        i = i + 1

        f1a, eta = tidal_heat(a, nm, eccm, parameters)
        f1c, eta = tidal_heat(c, nm, eccm, parameters)
        f1da, eta = tidal_heat((a + 0.001), nm, eccm, parameters)
        f1dc, eta = tidal_heat((c + 0.001), nm, eccm, parameters)

        df1a = (f1da - f1a) / (a + 0.001 - a)
        df1c = (f1dc - f1c) / (c + 0.001 - c)

        if (df1a * df1c) < 0:
            b = c
            c = (a + b) / 2.

        elif (df1a * df1c) > 0:
            a = c
            c = (a + b) / 2.

        elif (df1a * df1c) == 0:
            if df1a == 0:
                peak = a
            if df1c == 0:
                peak = c

    if b == T_breakdown:
        T_equilibrium = -5
        error = True
    else:
        f1a, eta = tidal_heat(a, nm, eccm, parameters)
        f1b, eta = tidal_heat(b, nm, eccm, parameters)
        peak = (a + b) / 2.

    if error:
        if T_equilibrium == -5:
            print("___error5___: no peak of tidal heat flux??")
            return T_equilibrium

    # Find T_stab between the peak and b0:
    a = peak
    b = b0
    un1 = 0
    un2 = 0

    T_equilibrium, a, b, again, unstable = intersection(a, b, nm, eccm, parameters)

    if T_equilibrium == -3:
        return T_equilibrium

    un1 = unstable

    if T_equilibrium != 0:
        return T_equilibrium
    elif b == b0:    # T_equilibrium not found or does not exist
        b = peak
        a = a0
        again = True    # try again below the peak
    elif un1 == 0:
        f1a, eta = tidal_heat(a, nm, eccm, parameters)
        f2a = convection_heat(a, eta, parameters)

        f1b, eta = tidal_heat(b, nm, eccm, parameters)
        f2b = convection_heat(b, eta, parameters)

        if f2a == 0 or f2b == 0:
            error = True
            T_equilibrium = -1

        stab = check((f1b - f2b), (f1a - f2a))

        if stab:
            T_equilibrium = (a + b) / 2.    # stable point (T_stab) is found
            return T_equilibrium
        elif not stab:    # unstable point is found
            a = b
            b = b0
            un1 = un1 + 1
            again = True    # try again above the unstable point
        else:
            T_equilibrium = -3
            error = True

    if error:
        if T_equilibrium == -3:
            print("___error3___: T_equilibrium not found??")
        return T_equilibrium

    # Search for T_stab again (below the peak or above the unstable point)
    if again:

        T_equilibrium, a, b, again, unstable = intersection(a, b, nm, eccm, parameters)

        if T_equilibrium == -3:
            return T_equilibrium

        un2 = unstable

        if T_equilibrium != 0:
            return T_equilibrium
        elif b == b0:    # T_unstab is found, but T_stab is not found
            return T_equilibrium  # This is because the unstable and stable points are too close
        elif b == peak:    # T_equilibrium does not exist
            return T_equilibrium
        elif un2 == 0:
            f1a, eta = tidal_heat(a, nm, eccm, parameters)
            f2a = convection_heat(a, eta, parameters)

            f1b, eta = tidal_heat(b, nm, eccm, parameters)
            f2b = convection_heat(b, eta, parameters)

            if f2a == 0 or f2b == 0:
                error = True
                T_equilibrium = -1

            stab = check((f1b - f2b), (f1a - f2a))

            if stab:
                T_equilibrium = (a + b) / 2.  # stable point (T_stab) is found
                return T_equilibrium
            elif not stab:  # unstable point is found
                if un1 == 0:
                    a = b
                    b = peak
                    un2 = un2 + 1
                    again = True  # try again above the unstable point
                else:
                    T_equilibrium = -2
                    error = True
            else:
                T_equilibrium = -3
                error = True

        # T_equilibrium does not exist (no intersection)
        if un1 == 0 and un2 == 0 and T_equilibrium == 0:
            return T_equilibrium

        if error:
            if T_equilibrium == -2:
                print("___error2___: two unstable equilibrium points??")
            if T_equilibrium == -3:
                print("___error3___: T_equilibrium not found??")
            return T_equilibrium

        if again:
            T_equilibrium, a, b, again, unstable = intersection(a, b, nm, eccm, parameters)

            if T_equilibrium == -3:
                return T_equilibrium

            un3 = unstable

            if T_equilibrium != 0:
                return T_equilibrium
            # T_unstab is found, but T_stab is not found (or does not exist?)
            elif b == peak:
                T_equilibrium = -6
                error = True
            elif un3 == 0:
                f1a, eta = tidal_heat(a, nm, eccm, parameters)
                f2a = convection_heat(a, eta, parameters)

                f1b, eta = tidal_heat(b, nm, eccm, parameters)
                f2b = convection_heat(b, eta, parameters)

                if f2a == 0 or f2b == 0:
                    error = True
                    T_equilibrium = -1

                stab = check((f1b - f2b), (f1a - f2a))

                if stab:
                    T_equilibrium = (a + b) / 2.    # stable point (T_stab) is found
                    return T_equilibrium
                elif not stab:    # unstable point is found
                    T_equilibrium = -2
                    error = True    # two unstable points
                else:
                    T_equilibrium = -3
                    error = True

            if error:
                if T_equilibrium == -2:
                    print("___error2___: two unstable equilibrium points??")
                if T_equilibrium == -3:
                    print("___error3___: T_equilibrium not found??")
                if T_equilibrium == -6:
                    print("___error6___: T_unstab is found, but T_stab is not found")
                return T_equilibrium

    print(T_equilibrium)
    return T_equilibrium
