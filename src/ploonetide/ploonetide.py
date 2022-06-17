"""This module defines TidalSimulation class"""
from __future__ import division
import os
# import logging

import pandas as pd
import numpy as np
import pyfiglet

from . import PACKAGEDIR
from ploonetide.utils.constants import MSUN, RSUN, GYEAR, DAY, PLANETS
from ploonetide.utils.functions import *
from ploonetide.odes.planet_moon import solution_planet_moon
from ploonetide.odes.star_planet import solution_star_planet
from ploonetide.forecaster.mr_forecast import Mstat2R
from ploonetide.numerical.simulator import Variable, Simulation

__all__ = ['TidalSimulation']


class TidalSimulation(Simulation):
    """This class defines a tidal simulation.

    Attributes:
        Args:
            moon_albedo (float, optional): Moon albedo [No unit]
            moon_eccentricty (float, optional): Eccentricity of moon's orbit [No unit]
            moon_mass (int, optional): Moon mass [Mearth]
            moon_radius (int, optional): Moon radius [Rearth]
            moon_rotperiod (float): Rotation period of the moon [s]
            moon_temperature (float): Temperature of the moon [K]
            planet_alpha (float, optional): Planet's radius aspect ratio [No unit]
            planet_angular_coeff (float, optional): Planet's mass fraction involved in angular momentum exchange [No unit]
            planet_beta (float, optional): Planet's mass aspect ratio [No unit]
            planet_eccentricity (float, optional): Planetary eccentricty [No unit]
            planet_mass (int, optional): Planetary mass [Mjup]
            planet_orbperiod (None, optional): Planetary orbital period [d]
            planet_radius (None, optional): Planetary radius [Rjup]
            planet_rigidity (float, optional): Rigidity of the planet [Pa]
            planet_roche_radius (float): Roche radius of the planet [m]
            planet_rotperiod (float, optional): Planetary rotation period [d]
            star_age (int, optional): Stellar age [Gyr]
            star_alpha (float, optional): Stellar radius aspect ratio [No unit]
            star_angular_coeff (float, optional): Star's fraction of mass partaking in angular momentum exchange [No unit]
            star_beta (float, optional): Stellar mass aspect ratio [No unit]
            star_eff_temperature (int, optional): Stellar effective temperature [K]
            star_mass (int, optional): Stellar mass [Msun]
            star_radius (int, optional): Stellar radius [Rsun]
            star_rotperiod (int, optional): Stellar rotation period [d]
            star_saturation_rate (float, optional): Star's saturation rotational rate [rad s^-1]
            sun_mass_loss_rate (float, optional): Solar mass loss rate [Msun yr^-1]
            sun_omega (float, optional): Solar rotational rate [s^-1]
            system (str, optional): Flag to choose type of system. Either 'star-planet' or 'planet-moon'

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
        planet_semima (float): Semi-major axis of the planet [m]
        star_alpha (float, optional): Stellar radius aspect ratio [No unit]
        star_angular_coeff (float, optional): Star's fraction of mass partaking in angular momentum exchange [No unit]
        star_beta (float, optional): Stellar mass aspect ratio [No unit]
        star_eff_temperature (int, optional): Stellar effective temperature [K]
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
                 moon_radius=1, moon_mass=1, moon_albedo=0.3, moon_eccentricty=0.02,
                 moon_semimaxis=10, system='star-planet'):
        """Construct the class

        Args:
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
        # SET CANONICAL UNITS
        # ************************************************************
        # self.mass_unit = mass_unit
        # self.length_unit = length_unit
        # self.time_unit = time_unit
        # self.uM, self.uL, self.uT = canonic_units(uL=self.length_unit, uM=self.mass_unit, uT=self.time_unit)

        # General constants
        self.sun_mass_loss_rate = sun_mass_loss_rate * MSUN / YEAR
        self.sun_omega = sun_omega

        # ************************************************************
        # STAR PARAMETERS
        # ************************************************************
        self.star_mass = star_mass * MSUN
        self.star_radius = star_radius * RSUN
        self.star_eff_temperature = star_eff_temperature
        self.star_rotperiod = star_rotperiod * DAY
        self.star_saturation_rate = star_saturation_rate
        self.star_saturation_period = 2. * np.pi / self.star_saturation_rate
        self.star_angular_coeff = star_angular_coeff
        self.star_alpha = star_alpha
        self.star_beta = star_beta
        self.star_omega = 2. * np.pi / self.star_rotperiod
        self.star_epsilon = self.star_omega / omegaCritic(self.star_mass, self.star_radius)
        self.star_k2q = k2Q_star_envelope(self.star_alpha, self.star_beta, self.star_epsilon)
        self.star_age = star_age * GYEAR
        self.star_luminosity = luminosity(self.star_radius, self.star_eff_temperature)

        self.stellar_lifespan = stellar_lifespan(self.star_mass)

        # ************************************************************
        # PLANET PARAMETERS
        # ************************************************************
        self.planet_rigidity = planet_rigidity
        self.planet_angular_coeff = planet_angular_coeff
        self.planet_mass = planet_mass * PLANETS.Jupiter.M
        if pd.isnull(planet_radius):
            self.planet_radius, _, _ = Mstat2R(mean=self.planet_mass / PLANETS.Jupiter.M,
                                               std=0.1, unit='Jupiter', sample_size=200,
                                               classify='Yes')
            self.planet_radius = self.planet_radius * PLANETS.Jupiter.R
        else:
            self.planet_radius = planet_radius * PLANETS.Jupiter.R

        self.planet_orbperiod = planet_orbperiod * DAY
        self.planet_rotperiod = planet_rotperiod * DAY
        self.planet_eccentricity = planet_eccentricity
        self.planet_omega = 2. * np.pi / self.planet_rotperiod
        self.planet_alpha = planet_alpha
        self.planet_beta = planet_beta
        self.planet_semima = semiMajorAxis(self.planet_orbperiod, self.star_mass, self.planet_mass)
        self.planet_meanmo = meanMotion(self.planet_semima, self.star_mass, self.planet_mass)
        self.planet_epsilon = self.planet_omega / omegaCritic(self.planet_mass, self.planet_radius)
        self.planet_k2q = k2Q_planet_envelope(self.planet_alpha, self.planet_beta, self.planet_epsilon)
        if self.__planet_core_dissipation:
            self.planet_k2q = k2Q_planet_envelope(self.planet_alpha, self.planet_beta, self.planet_epsilon) +\
                k2Q_planet_core(self.planet_rigidity, self.planet_alpha, self.planet_beta,
                                self.planet_mass, self.planet_radius)

        self.planet_roche_radius = 2.7 * (self.star_mass / self.planet_mass)**(1. / 3.) * self.planet_radius  # Roche radius of the planet (Guillochon et. al 2011)

        # ************************************************************
        # MOON PARAMETERS
        # ************************************************************
        self.moon_eccentricty = moon_eccentricty
        self.moon_albedo = moon_albedo
        self.moon_mass = moon_mass * MEARTH
        self.moon_radius = moon_radius * REARTH
        self.moon_roche_radius = aRoche_solid(self.planet_mass, self.moon_mass, self.moon_radius)
        self.moon_semimaxis = moon_semimaxis * self.moon_roche_radius
        self.moon_meanmo = meanMotion(self.moon_semimaxis, self.planet_mass, self.moon_mass)
        self.moon_rotperiod = 2. * np.pi / self.moon_meanmo
        self.moon_density = density(self.moon_mass, self.moon_radius)
        self.moon_temperature = equil_temp(self.star_eff_temperature, self.star_radius,
                                           self.planet_semima, self.moon_albedo)
        self.moon_tidal_ene = e_tidal(self.moon_temperature, self.moon_meanmo,
                                      densm=self.moon_density, Mm=self.moon_mass,
                                      Rm=self.moon_radius, eccm=self.moon_eccentricty)

        # Arguments for including/excluding different effects
        self.args = dict(
            star_internal_evolution=self._star_internal_evolution, star_k2q=self.star_k2q,
            planet_internal_evolution=self._planet_internal_evolution, planet_k2q=self.planet_k2q,
            planet_size_evolution=self._planet_size_evolution, Rp=self.planet_radius,
            planet_core_dissipation=self._planet_core_dissipation,
        )

        # Parameters dictionary
        self.parameters = dict(Ms=self.star_mass, Rs=self.star_radius, Ls=self.star_luminosity,
                               coeff_star=self.star_angular_coeff, star_alpha=self.star_alpha,
                               star_beta=self.star_beta, os_saturation=self.star_saturation_rate,
                               star_age=self.star_age, coeff_planet=self.planet_angular_coeff,
                               Mp=self.planet_mass, Rp=self.planet_radius, planet_alpha=self.planet_alpha,
                               planet_beta=self.planet_beta, rigidity=self.planet_rigidity,
                               E_act=activation_energy, B=melt_fraction_coeff,
                               Ts=solidus_temperature, Tb=breakdown_temperature,
                               Tl=liquidus_temperature, Cp=heat_capacity,
                               ktherm=thermal_conductivity, Rac=Rayleigh_critical,
                               a2=flow_geometry, alpha_exp=thermal_expansivity,
                               densm=self.moon_density, Mm=self.moon_mass, Rm=self.moon_radius,
                               melt_fr=self.melt_fraction, sun_omega=self.sun_omega,
                               sun_mass_loss_rate=self.sun_mass_loss_rate, args=self.args)

        # ************************************************************
        # INITIAL CONDITIONS FOR THE SYSTEM
        # ************************************************************
        if self.system == 'star-planet':
            self.parameters["om_ini"] = self.planet_omega  # Initial planet's rotational rate
            self.parameters["e_ini"] = self.planet_eccentricity  # Initial eccentricity
            self.parameters["os_ini"] = self.star_omega  # Initial star's rotational rate
            self.parameters["npp_ini"] = self.planet_meanmo  # Initial planet mean motion
            self.parameters["mp_ini"] = self.planet_mass  # Initial planetary mass

            motion_p = Variable('planet_mean_motion', self.planet_meanmo)
            omega_p = Variable('planet_omega', self.planet_omega)
            eccen_p = Variable('planet_eccentricity', self.planet_eccentricity)
            omega_s = Variable('star_omega', self.star_omega)
            mass_p = Variable('planet_mass', self.planet_mass)
            initial_variables = [motion_p, omega_p, eccen_p, omega_s, mass_p]

            print(f'\nStellar mass: {self.star_mass / MSUN:.1f} Msun\n',
                  f'Planet orbital period: {self.planet_orbperiod / DAY:.1f} days\n',
                  f'Planetary mass: {self.planet_mass / PLANETS.Jupiter.M:.1f} Mjup\n',
                  f'Planetary radius: {self.planet_radius / PLANETS.Jupiter.R:.1f} Rjup\n')

        elif self.system == 'planet-moon':
            self.parameters['nm_ini'] = self.moon_meanmo  # Moon's initial mean motion
            self.parameters['np_ini'] = self.planet_meanmo  # PLanet's initial mean motion
            self.parameters['op_ini'] = self.planet_omega   # Planet's initial rotation rate
            self.parameters['Tm_ini'] = self.moon_temperature  # Moon's initial temperature
            self.parameters['Em_ini'] = self.moon_tidal_ene  # Moon's initial tidal heat
            self.parameters['em_ini'] = self.moon_eccentricty  # Moon's initial eccentricity

            motion_m = Variable('mean_motion_m', self.moon_meanmo)
            omega_p = Variable('omega_planet', self.planet_omega)
            motion_p = Variable('mean_motion_p', self.planet_meanmo)
            temper_m = Variable('temperature', self.moon_temperature)
            tidal_m = Variable('tidal_heat', self.moon_tidal_ene)
            eccen_m = Variable('eccentricity', self.moon_eccentricty)
            initial_variables = [motion_m, omega_p, motion_p, temper_m, tidal_m, eccen_m]
            if self.parameters['em_ini'] == 0.0:
                initial_variables = [motion_m, omega_p, motion_p, temper_m, tidal_m]

            print(f'\nStellar mass: {self.star_mass / MSUN:.1f} Msun\n',
                  f'Planet orbital period: {self.planet_orbperiod / DAY:.1f} days\n',
                  f'Planetary mass: {self.planet_mass / PLANETS.Jupiter.M:.1f} Mjup\n',
                  f'Planetary radius: {self.planet_radius / PLANETS.Jupiter.R:.1f} Rjup\n',
                  f'Moon orbital period: {moon_semimaxis:.1f} a_roche ({self.moon_rotperiod / DAY:.1f} days)\n')

        super().__init__(variables=initial_variables)

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
