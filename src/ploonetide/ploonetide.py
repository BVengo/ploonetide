"""This module defines TidalSimulation class"""
from __future__ import division
import os
# import logging

import astropy.units as u
import pandas as pd
import numpy as np
import pyfiglet

from . import PACKAGEDIR
from ploonetide.utils.constants import PLANETS
from ploonetide.utils.functions import *
from ploonetide.odes.planet_moon import solution_planet_moon
from ploonetide.odes.star_planet import solution_star_planet
from ploonetide.forecaster.mr_forecast import Mstat2R
from ploonetide.numerical.simulator import Variable, Simulation

__all__ = ['TidalSimulation']


class TidalSimulation(Simulation):
    """This class defines a tidal simulation.

    Attributes:
        system (str, optional): Flag to choose type of system. Either 'star-planet' or 'planet-moon'
        moon_albedo (float, optional): Moon albedo [No unit]
        moon_temperature (float): Temperature of the moon [K]
        planet_alpha (float, optional): Planet's radius aspect ratio [No unit]
        planet_angular_coeff (float, optional): Planet's mass fraction for angular momentum exchange [No unit]
        planet_beta (float, optional): Planet's mass aspect ratio [No unit]
        planet_roche_radius (float): Roche radius of the planet [m]
        planet_rotperiod (float, optional): Planetary rotation period [d]
        star_alpha (float, optional): Stellar radius aspect ratio [No unit]
        star_angular_coeff (float, optional): Star's mass fraction for angular momentum exchange [No unit]
        star_beta (float, optional): Stellar mass aspect ratio [No unit]
        moon_density (float): Density of the moon [kg * m^-3]
        moon_meanmo (float): Initial mean motion of the moon [s^-1]
        moon_radius (int, optional): Moon radius [Rearth]
        moon_roche_radius (float): Roche radius of the moon [m]
        moon_semimaxis (None, optional): Moon's semi-major axis [a_Roche]
        moon_temperature (float): Temperature of the moon [K]
        moon_tidal_ene (float): Tidal energy of the moon [J]
        planet_epsilon (float): Epsilon rate of the planet [s^-1]
        planet_k2q (float): Tidal heat function of the planet [J^-1]
        planet_meanmo (float): Initial mean motion of the planet [s^-1]
        planet_omega (float): Initial rotational rate of the planet [s^-1]
        planet_roche_radius (float): Roche radius of the planet [m]
        planet_semimaxis (float): Semi-major axis of the planet [m]
        star_alpha (float, optional): Stellar radius aspect ratio [No unit]
        star_beta (float, optional): Stellar mass aspect ratio [No unit]
        star_epsilon (float): Description
        star_k2q (float): Tidal heat function of the star [J^-1]
        star_luminosity (float): Stellar luminosity [W]
        star_omega (float): Description
        star_saturation_period (float): Saturation period for the stellar rotation [s]
        stellar_lifespan (float): Lifespan of the star
    """

    def __init__(self, activation_energy=3E5, melt_fraction=0.5, heat_capacity=1260,
                 melt_fraction_coeff=25., solidus_temperature=1600., breakdown_temperature=1800.,
                 liquidus_temperature=2000., thermal_conductivity=2., Rayleigh_critical=1100.,
                 flow_geometry=1., thermal_expansivity=1E-4, planet_size_evolution=False,
                 planet_internal_evolution=False, planet_core_dissipation=False,
                 star_internal_evolution=False, star_mass=1., star_radius=1.,
                 star_eff_temperature=3700., star_saturation_rate=4.3421E-5,
                 star_angular_coeff=0.5, star_rotperiod=10, star_alpha=0.25, star_beta=0.25,
                 star_age=5., sun_omega=2.67E-6, sun_mass_loss_rate=1.4E-14, planet_mass=1.,
                 planet_radius=None, planet_angular_coeff=0.26401, planet_orbperiod=None,
                 planet_rotperiod=0.6, planet_eccentricity=0.1, planet_rigidity=4.46E10,
                 planet_alpha=PLANETS.Jupiter.alpha, planet_beta=PLANETS.Jupiter.beta,
                 moon_radius=1, moon_mass=1, moon_albedo=0.3, moon_eccentricity=0.02,
                 moon_semimaxis=10, system='star-planet'):
        """Construct the class

        Args (and attributes):
            activation_energy (float, optional): Energy of activation, default is 3e5 [J mol^-1]
            melt_fraction (float, optional): Fraction of melt for ice, default is 0.5 [No unit]
            heat_capacity (int, optional): Heat capacity of moon material, default is 1260 [J kg^-1 K^-1]
            melt_fraction_coeff (int, optional): Coefficient for melt fraction, default is 25 [No unit]
            solidus_temperature (int, optional): Temperature for solid material, default is 1600 [K]
            breakdown_temperature (int, optional): Temperature of breakdown from solid to liquidus, default is 1800 [K]
            liquidus_temperature (int, optional): Temperature for liquid material, default is 2000 [K]
            thermal_conductivity (int, optional): Description, default is 2 [W m^-1 K^-1]
            Rayleigh_critical (int, optional): Critical rayleigh number, default is 1100 [No unit]
            flow_geometry (int, optional): Constant for flow geometry [No unit]
            thermal_expansivity (float, optional): Thermal expansivity of the moon, default is 1E-4 [K^-1]
            sun_mass_loss_rate (float, optional): Solar mass loss rate [Msun yr^-1]
            star_rotperiod (int, optional): Stellar rotation period [d]
            star_saturation_rate (float, optional): Star's saturation rotational rate [rad s^-1]
            sun_omega (float, optional): Solar rotational rate [s^-1]
            star_age (int, optional): Stellar age [Gyr]
            star_eff_temperature (int, optional): Stellar effective temperature [K]
            star_mass (int, optional): Stellar mass [Msun]
            star_radius (int, optional): Stellar radius [Rsun]
            planet_eccentricity (float, optional): Planetary eccentricity [No unit]
            planet_mass (int, optional): Planetary mass [Mjup]
            planet_orbperiod (None, optional): Planetary orbital period [d]
            planet_radius (None, optional): Planetary radius [Rjup]
            planet_rigidity (float, optional): Rigidity of the planet [Pa]
            moon_mass (int, optional): Moon mass [Mearth]
            moon_radius (int, optional): Moon radius [Rearth]
            moon_rotperiod (float): Rotation period of the moon [s]
            moon_eccentricity (float, optional): Eccentricity of moon's orbit [No unit]
        """

        print(pyfiglet.figlet_format(f'{self.package}'))

        # ************************************************************
        # SET THE TYPE OF SYSTEM
        # ************************************************************
        self.system = system

        # ************************************************************
        # KEY TO INCLIDE EVOLUTION
        # ************************************************************
        self._planet_size_evolution = planet_size_evolution
        self._planet_internal_evolution = planet_internal_evolution
        self._planet_core_dissipation = planet_core_dissipation
        self._star_internal_evolution = star_internal_evolution

        # ************************************************************
        # GENERAL CONSTANTS IN THE SIMULATION
        # ************************************************************
        self._sun_mass_loss_rate = u.Quantity(sun_mass_loss_rate, u.Msun * u.yr**-1)
        self._sun_omega = u.Quantity(sun_omega, u.s**-1)
        self._activation_energy = u.Quantity(activation_energy, u.J * u.mol**-1)
        self._solidus_temperature = u.Quantity(solidus_temperature, u.K)
        self._breakdown_temperature = u.Quantity(breakdown_temperature, u.K)
        self._liquidus_temperature = u.Quantity(liquidus_temperature, u.K)
        self._heat_capacity = u.Quantity(heat_capacity, u.J * u.kg**-1 * u.K**-1)
        self._thermal_conductivity = u.Quantity(thermal_conductivity, u.W * u.m**-1 * u.K**-1)
        self._thermal_expansivity = u.Quantity(thermal_expansivity, u.K**-1)
        self.Rayleigh_critical = Rayleigh_critical
        self.flow_geometry = flow_geometry
        self.melt_fraction_coeff = melt_fraction_coeff
        self.melt_fraction = melt_fraction

        # ************************************************************
        # STAR PARAMETERS
        # ************************************************************
        self._star_mass = u.Quantity(star_mass, u.Msun)
        self._star_radius = u.Quantity(star_radius, u.Rsun)
        self._star_eff_temperature = u.Quantity(star_eff_temperature, u.K)
        self._star_rotperiod = u.Quantity(star_rotperiod, u.d)
        self._star_age = u.Quantity(star_age, u.Gyr)
        self._star_saturation_rate = u.Quantity(star_saturation_rate, u.s**-1)
        self.star_angular_coeff = star_angular_coeff
        self.star_alpha = star_alpha
        self.star_beta = star_beta

        # ************************************************************
        # PLANET PARAMETERS
        # ************************************************************
        self._planet_orbperiod = u.Quantity(planet_orbperiod, u.d)
        self._planet_rotperiod = u.Quantity(planet_rotperiod, u.d)
        self._planet_mass = u.Quantity(planet_mass, u.M_jup)
        self._planet_radius = u.Quantity(planet_radius, u.R_jup)
        self._planet_rigidity = u.Quantity(planet_rigidity, u.Pa)
        self.planet_angular_coeff = planet_angular_coeff

        self.planet_eccentricity = planet_eccentricity
        self.planet_alpha = planet_alpha
        self.planet_beta = planet_beta

        # ************************************************************
        # MOON PARAMETERS
        # ************************************************************
        self._moon_mass = u.Quantity(moon_mass, u.Mearth)
        self._moon_radius = u.Quantity(moon_radius, u.Rearth)
        self._moon_semimaxis = u.Quantity(moon_semimaxis * self.moon_roche_radius.value, u.m)
        self.moon_eccentricity = moon_eccentricity
        self.moon_albedo = moon_albedo

        # Arguments for including/excluding different effects
        self.args = dict(
            star_internal_evolution=self._star_internal_evolution, star_k2q=self.star_k2q,
            planet_internal_evolution=self._planet_internal_evolution, planet_k2q=self.planet_k2q,
            planet_size_evolution=self._planet_size_evolution, Rp=self.planet_radius.to(u.m).value,
            planet_core_dissipation=self._planet_core_dissipation,
        )

        # ************************************************************
        # INITIAL CONDITIONS FOR THE SYSTEM
        # ************************************************************
        if self.system == 'star-planet':
            motion_p = Variable('planet_mean_motion', self.planet_meanmo.value)
            omega_p = Variable('planet_omega', self.planet_omega.value)
            eccen_p = Variable('planet_eccentricity', self.planet_eccentricity)
            omega_s = Variable('star_omega', self.star_omega.value)
            mass_p = Variable('planet_mass', self.planet_mass.to(u.kg).value)
            initial_variables = [motion_p, omega_p, eccen_p, omega_s, mass_p]

            print(f'\nStellar mass: {self.star_mass:.1f} Msun\n',
                  f'Planet orbital period: {self.planet_orbperiod:.1f} days\n',
                  f'Planetary mass: {self.planet_mass:.1f} Mjup\n',
                  f'Planetary radius: {self.planet_radius.value:.1f} Rjup\n')

        elif self.system == 'planet-moon':
            omega_p = Variable('omega_planet', self.planet_omega.value)
            motion_p = Variable('mean_motion_p', self.planet_meanmo.value)
            motion_m = Variable('mean_motion_m', self.moon_meanmo.value)
            temper_m = Variable('temperature', self.moon_temperature.value)
            tidal_m = Variable('tidal_heat', self.moon_tidal_ene.value)
            eccen_m = Variable('eccentricity', self.moon_eccentricity)
            initial_variables = [omega_p, motion_p, motion_m, temper_m, tidal_m, eccen_m]
            if self.moon_eccentricity == 0.0:
                initial_variables = [omega_p, motion_p, motion_m, temper_m, tidal_m]

            print(f'\nStar mass: {self.star_mass:.1f}\n',
                  f'Star radius: {self.star_radius:.1f}\n',
                  f'Star rotation period: {self.star_rotperiod:.1f}\n',
                  f'Planet orbital period: {self.planet_orbperiod:.1f}\n',
                  f'Planet mass: {self.planet_mass:.1f}\n',
                  f'Planet radius: {self.planet_radius:.1f}\n',
                  f'Planet eccentricity: {self.planet_eccentricity:.1f}\n',
                  f'Moon mass: {self.moon_mass:.1f}\n',
                  f'Moon radius: {self.moon_radius:.1f}\n',
                  f'Moon eccentricity: {self.moon_eccentricity:.1f}\n',
                  f'Moon orbital period: {moon_semimaxis:.1f} a_roche ({self.moon_orbperiod:.1f})\n')

        super().__init__(variables=initial_variables)

    @property
    def parameters(self):
        # Parameters dictionary of the simulation
        return dict(
            Ms=self.star_mass.to(u.kg).value, Rs=self.star_radius.to(u.m).value,
            Ls=self.star_luminosity.value, coeff_star=self.star_angular_coeff,
            star_alpha=self.star_alpha, star_beta=self.star_beta,
            os_saturation=self.star_saturation_rate.value, star_age=self.star_age.to(u.s).value,
            coeff_planet=self.planet_angular_coeff, Mp=self.planet_mass.to(u.kg).value,
            Rp=self.planet_radius.to(u.m).value, planet_alpha=self.planet_alpha,
            planet_beta=self.planet_beta, rigidity=self.planet_rigidity.value,
            E_act=self.activation_energy.value, B=self.melt_fraction_coeff,
            Ts=self.solidus_temperature.value, Tb=self.breakdown_temperature.value,
            Tl=self.liquidus_temperature.value, Cp=self.heat_capacity.value,
            ktherm=self.thermal_conductivity.value, Rac=self.Rayleigh_critical,
            a2=self.flow_geometry, alpha_exp=self.thermal_expansivity.value,
            densm=self.moon_density.value, Mm=self.moon_mass.to(u.kg).value,
            Rm=self.moon_radius.to(u.m).value, melt_fr=self.melt_fraction,
            sun_mass_loss_rate=self.sun_mass_loss_rate.to(u.kg * u.s**-1).value,
            sun_omega=self.sun_omega.value, os_ini=self.star_omega.value,
            np_ini=self.planet_meanmo.value, op_ini=self.planet_omega.value,
            ep_ini=self.planet_eccentricity, mp_ini=self.planet_mass.to(u.kg).value,
            nm_ini=self.moon_meanmo.value, Tm_ini=self.moon_temperature.value,
            Em_ini=self.moon_tidal_ene.value, em_ini=self.moon_eccentricity,
            args=self.args
        )

    # **********************************************************************************************
    # ******************************* GENERAL CONSTANTS MODIFIABLE *********************************
    # **********************************************************************************************

    @property
    def sun_mass_loss_rate(self):
        return self._sun_mass_loss_rate

    @sun_mass_loss_rate.setter
    def sun_mass_loss_rate(self, value):
        self._sun_mass_loss_rate = value
        if not isinstance(self._sun_mass_loss_rate, u.Quantity):
            self._sun_mass_loss_rate = u.Quantity(value, u.Msun * u.yr**-1)

    @property
    def sun_omega(self):
        return self._sun_omega

    @sun_omega.setter
    def sun_omega(self, value):
        self._sun_omega = value
        if not isinstance(self._sun_omega, u.Quantity):
            self._sun_omega = u.Quantity(value, u.s**-1)

    @property
    def activation_energy(self):
        return self._activation_energy

    @activation_energy.setter
    def activation_energy(self, value):
        self._activation_energy = value
        if not isinstance(self._activation_energy, u.Quantity):
            self._activation_energy = u.Quantity(value, u.J * u.mol**-1)

    @property
    def solidus_temperature(self):
        return self._solidus_temperature

    @solidus_temperature.setter
    def solidus_temperature(self, value):
        self._solidus_temperature = value
        if not isinstance(self._solidus_temperature, u.Quantity):
            self._solidus_temperature = u.Quantity(value, u.K)

    @property
    def liquidus_temperature(self):
        return self._liquidus_temperature

    @liquidus_temperature.setter
    def liquidus_temperature(self, value):
        self._liquidus_temperature = value
        if not isinstance(self._liquidus_temperature, u.Quantity):
            self._liquidus_temperature = u.Quantity(value, u.K)

    @property
    def breakdown_temperature(self):
        return self._breakdown_temperature

    @breakdown_temperature.setter
    def breakdown_temperature(self, value):
        self._breakdown_temperature = value
        if not isinstance(self._breakdown_temperature, u.Quantity):
            self._breakdown_temperature = u.Quantity(value, u.K)

    @property
    def heat_capacity(self):
        return self._heat_capacity

    @heat_capacity.setter
    def heat_capacity(self, value):
        self._heat_capacity = value
        if not isinstance(self._heat_capacity, u.Quantity):
            self._heat_capacity = u.Quantity(value, u.J * u.kg**-1 * u.K**-1)

    @property
    def thermal_conductivity(self):
        return self._thermal_conductivity

    @thermal_conductivity.setter
    def thermal_conductivity(self, value):
        self._thermal_conductivity = value
        if not isinstance(self._thermal_conductivity, u.Quantity):
            self._thermal_conductivity = u.Quantity(value, u.W * u.m**-1 * u.K**-1)

    @property
    def thermal_expansivity(self):
        return self._thermal_expansivity

    @thermal_expansivity.setter
    def thermal_expansivity(self, value):
        self._thermal_expansivity = value
        if not isinstance(self._thermal_expansivity, u.Quantity):
            self._thermal_expansivity = u.Quantity(value, u.K**-1)

    # **********************************************************************************************
    # ********************************* STAR DYNAMICAL PROPERTIES **********************************
    # **********************************************************************************************
    @property
    def star_mass(self):
        return self._star_mass

    @star_mass.setter
    def star_mass(self, value):
        self._star_mass = value
        if not isinstance(self._star_mass, u.Quantity):
            self._star_mass = u.Quantity(value, u.Msun)

    @property
    def star_radius(self):
        return self._star_radius

    @star_radius.setter
    def star_radius(self, value):
        self._star_radius = value
        if not isinstance(self._star_radius, u.Quantity):
            self._star_radius = u.Quantity(value, u.Rsun)

    @property
    def star_age(self):
        return self._star_age

    @star_age.setter
    def star_age(self, value):
        self._star_age = value
        if not isinstance(self._star_age, u.Quantity):
            self._star_age = u.Quantity(value, u.Gyr)

    @property
    def star_saturation_rate(self):
        return self._star_saturation_rate

    @star_saturation_rate.setter
    def star_saturation_rate(self, value):
        self._star_saturation_rate = value
        if not isinstance(self._star_saturation_rate, u.Quantity):
            self._star_saturation_rate = u.Quantity(value, u.s**-1)

    @property
    def star_rotperiod(self):
        return self._star_rotperiod

    @star_rotperiod.setter
    def star_rotperiod(self, value):
        self._star_rotperiod = value
        if not isinstance(self._star_rotperiod, u.Quantity):
            self._star_rotperiod = u.Quantity(value, u.d)

    @property
    def star_eff_temperature(self):
        return self._star_eff_temperature

    @star_eff_temperature.setter
    def star_eff_temperature(self, value):
        self._star_eff_temperature = value
        if not isinstance(self._star_eff_temperature, u.Quantity):
            self._star_eff_temperature = u.Quantity(value, u.K)

    @property
    def star_luminosity(self):
        return u.Quantity(luminosity(self.star_radius.to(u.m).value, self.star_eff_temperature.value), u.W)

    @star_luminosity.setter
    def star_luminosity(self, value):
        self._star_luminosity = value
        if not isinstance(self._star_luminosity, u.Quantity):
            print("popito")
            self._star_luminosity = u.Quantity(value, u.W)

    @property
    def star_rotperiod(self):
        return self._star_rotperiod

    @star_rotperiod.setter
    def star_rotperiod(self, value):
        self._star_rotperiod = value
        if not isinstance(self._star_rotperiod, u.Quantity):
            self._star_rotperiod = u.Quantity(value, u.d)

    @property
    def star_omega(self):
        return u.Quantity(2. * np.pi / self.star_rotperiod.to(u.s).value, u.s**-1)

    @property
    def star_epsilon(self):
        return self.star_omega.value / omegaCritic(self.star_mass.to(u.kg).value, self.star_radius.to(u.m).value)

    @property
    def star_k2q(self):
        return k2Q_star_envelope(self.star_alpha, self.star_beta, self.star_epsilon)

    @property
    def star_lifespan(self):
        return u.Quantity(stellar_lifespan(self.star_mass.to(u.kg).value), u.s)

    @property
    def star_saturation_period(self):
        return u.Quantity(2. * np.pi / self.star_saturation_rate.value, u.d)

    # **********************************************************************************************
    # ******************************** PLANET DYNAMICAL PROPERTIES *********************************
    # **********************************************************************************************
    @property
    def planet_orbperiod(self):
        return self._planet_orbperiod

    @planet_orbperiod.setter
    def planet_orbperiod(self, value):
        self._planet_orbperiod = value
        if not isinstance(self._planet_orbperiod, u.Quantity):
            self._planet_orbperiod = u.Quantity(value, u.d)

    @property
    def planet_rotperiod(self):
        return self._planet_rotperiod

    @planet_rotperiod.setter
    def planet_rotperiod(self, value):
        self._planet_rotperiod = value
        if not isinstance(self._planet_rotperiod, u.Quantity):
            self._planet_rotperiod = u.Quantity(value, u.d)

    @property
    def planet_mass(self):
        return self._planet_mass

    @planet_mass.setter
    def planet_mass(self, value):
        self._planet_mass = value
        if not isinstance(self._planet_mass, u.Quantity):
            self._planet_mass = u.Quantity(value, u.M_jup)

    @property
    def planet_radius(self):
        if pd.isnull(self._planet_radius):
            planet_radius, _, _ = Mstat2R(
                mean=self.planet_mass.value, std=0.1, unit='Jupiter',
                sample_size=200, classify='Yes'
            )

            return u.Quantity(planet_radius, u.R_jup)
        else:
            return self._planet_radius

    @planet_radius.setter
    def planet_radius(self, value):
        self._planet_radius = value
        if not isinstance(self._planet_radius, u.Quantity):
            if not value:
                self._planet_radius, _, _ = Mstat2R(
                    mean=self.planet_mass.value, std=0.1, unit='Jupiter',
                    sample_size=200, classify='Yes'
                )
                self._planet_radius = u.Quantity(self._planet_radius, u.R_jup)

    @property
    def planet_rigidity(self):
        return self._planet_rigidity

    @planet_rigidity.setter
    def planet_rigidity(self, value):
        self._planet_rigidity = value
        if not isinstance(self._planet_rigidity, u.Quantity):
            self._planet_rigidity = u.Quantity(value, u.Pa)

    @property
    def planet_omega(self):
        return u.Quantity(2. * np.pi / self.planet_rotperiod.to(u.s).value, u.s**-1)

    @property
    def planet_semimaxis(self):
        return u.Quantity(semiMajorAxis(self.planet_orbperiod.to(u.s).value,
                                        self.star_mass.to(u.kg).value,
                                        self.planet_mass.to(u.kg).value), u.m).to(u.au)

    @property
    def planet_meanmo(self):
        return u.Quantity(meanMotion(self.planet_semimaxis.to(u.m).value,
                                     self.star_mass.to(u.kg).value,
                                     self.planet_mass.to(u.kg).value), u.s**-1)

    @property
    def planet_epsilon(self):
        return self.planet_omega.value / omegaCritic(self.planet_mass.to(u.kg).value,
                                                     self.planet_radius.to(u.m).value)

    @property
    def planet_k2q(self):
        if self.__planet_core_dissipation:
            return k2Q_planet_envelope(self.planet_alpha, self.planet_beta, self.planet_epsilon) +\
                k2Q_planet_core(self.planet_rigidity.value, self.planet_alpha, self.planet_beta,
                                self.planet_mass.to(u.kg).value, self.planet_radius.to(u.m).value)
        else:
            return k2Q_planet_envelope(self.planet_alpha, self.planet_beta, self.planet_epsilon)

    @property
    def planet_roche_radius(self):
        return u.Quantity(2.7 * (self.star_mass.to(u.kg).value / self.planet_mass.to(u.kg).value)**(1. / 3.) * self.planet_radius.to(u.m).value, u.m).to(u.AU)  # Roche radius of the planet (Guillochon et. al 2011)

    @property
    def planet_hill_radius(self):
        return u.Quantity(hill_radius(self.planet_semimaxis.to(u.m).value, self.planet_eccentricity,
                                      self.planet_mass.to(u.kg).value, self.star_mass.to(u.kg).value), u.m).to(u.AU)

    # **********************************************************************************************
    # ******************************** MOON DYNAMICAL PROPERTIES ***********************************
    # **********************************************************************************************
    @property
    def moon_mass(self):
        return self._moon_mass

    @moon_mass.setter
    def moon_mass(self, value):
        self._moon_mass = value
        if not isinstance(self._moon_mass, u.Quantity):
            self._moon_mass = u.Quantity(value, u.Mearth)

    @property
    def moon_radius(self):
        return self._moon_radius

    @moon_radius.setter
    def moon_radius(self, value):
        self._moon_radius = value
        if not isinstance(self._moon_radius, u.Quantity):
            self._moon_radius = u.Quantity(value, u.Rearth)

    @property
    def moon_roche_radius(self):
        return u.Quantity(aRoche_solid(self.planet_mass.to(u.kg).value, self.moon_mass.to(u.kg).value, self.moon_radius.to(u.m).value), u.m)

    @property
    def moon_semimaxis(self):
        return self._moon_semimaxis

    @moon_semimaxis.setter
    def moon_semimaxis(self, value):
        self._moon_semimaxis = value
        if not isinstance(self._moon_semimaxis, u.Quantity):
            self._moon_semimaxis = u.Quantity(value * self.moon_roche_radius.value, u.m)

    @property
    def moon_meanmo(self):
        return u.Quantity(meanMotion(self.moon_semimaxis.value, self.planet_mass.to(u.kg).value,
                                     self.moon_mass.to(u.kg).value), u.s**-1)

    @property
    def moon_orbperiod(self):
        return u.Quantity(2. * np.pi / self.moon_meanmo.value, u.s).to(u.d)

    @property
    def moon_density(self):
        return u.Quantity(density(self.moon_mass.to(u.kg).value, self.moon_radius.to(u.m).value), u.kg * u.m**-3.)

    @property
    def moon_temperature(self):
        return u.Quantity(equil_temp(self.star_eff_temperature.value, self.star_radius.to(u.m).value,
                                     self.planet_semimaxis.to(u.m).value, self.moon_albedo), u.K)

    @property
    def moon_tidal_ene(self):
        return u.Quantity(e_tidal(self.moon_temperature.value, self.moon_meanmo.value,
                                  densm=self.moon_density.value, Mm=self.moon_mass.to(u.kg).value,
                                  Rm=self.moon_radius.to(u.m).value, eccm=self.moon_eccentricity), u.J * u.s**-1)

    @classmethod
    def get_class_name(cls):
        """Get the name TidalSimulation as a string.

        Returns:
            str: Name of the class
        """
        return cls.__name__

    @classmethod
    def __getattr__(self, name):
        return f'{self.get_class_name()} does not have "{str(name)}" attribute'

    @property
    def package(self):
        """Get the name of the package.

        Returns:
            str: Name of the package
        """
        return os.path.basename(PACKAGEDIR)

    def run(self, integration_time, timestep, t0=0):
        differential_equation = solution_star_planet
        if self.system == 'planet-moon':
            differential_equation = solution_planet_moon
        super().set_diff_eq(differential_equation, **self.parameters)
        return super().run(integration_time, timestep, t0=0)
