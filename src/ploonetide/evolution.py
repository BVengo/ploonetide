#!/usr/bin/env python
# -*- coding:utf-8 -*-
from ploonetide.odes.core import *
from simulator import Simulation, Variable
from exoplanets_df import df
import mr_forecast as forecaster

import subprocess
import matplotlib as mpl
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
# import plotly.figure_factory as ff
# import plotly.express as px
# import ternary
import numpy as np
import time
import sys
import pandas as pd

# ************************************************************
# SET UNITS
# ************************************************************
uM, uL, uT = canonic_units(uL=RSUN, uM=MSUN)

# General constants
E_act = 300 * 1000 * uM**-1. * uL**-2. * uT**2.  # J mol^-1  -->  kg * m^2 s^-2 * mol^-1
Rg = const.gas_constant * uM**-1. * uL**-2. * uT**2.  # J mol^-1 K^-1  -->  kg * m^2 s^-2 * mol^-1 * K-1
sigmasb = const.sigma * uM**-1 * uT**3.  # W m^-2 K^-4  -->  kg s^-3 K^-4
melt_fr = 0.5  # No unit
Cp = 1260 * uL**-2. * uT**2.  # J kg^-1 K^-1  -->  m^2 s^-2 K^-1
B = 25  # No unit
Ts = 1600  # K
Tb = 1800  # K
T1 = 2000  # K
ktherm = 2 * uM**-1 * uL**-1. * uT**3.  # W m^-1 K^-1  -->  kg m s^-3 K^-1
Rac = 1100  # No unit
a2 = 1  # No unit
alpha_exp = 1e-4  # K^-1

shear_Conv = uM**-1. * uL * uT**2.  # kg m^-1 s^-2
visc_Conv = shear_Conv * uT**-1.  # kg m^-1 s^-1

# ************************************************************
# GLOBAL PARAMETERS
# ************************************************************
Protini = 15.656 * HOUR / uT  # Initial planetary rotational period
opini = 2 * np.pi / Protini

alpha0 = 0.249
beta0 = 0.112

# ************************************************************
# MODIFICABLE PARAMETERS MOON
# ************************************************************
Mm = 1 * MEARTH * uM**-1  # Earth
Rm = 1 * REARTH * uL**-1  # Earth
Albem = 0.3
emini = 0.02
densm = density(Mm, Rm)

ind = 20
Mss = df["st_mass"][:ind].values * MSUN * uM**-1
Rss = df["st_rad"][:ind].values * RSUN * uL**-1
Teffs = df["st_teff"][:ind].values
Mps = df["pl_bmassj"][:ind].values * PLANETS.Jupiter.M * uM**-1
Rps = df["pl_radj"][:ind].values * PLANETS.Jupiter.R * uL**-1
Per = df["pl_orbper"][:ind].values * DAY * uT**-1

Mss = np.array([1.0]) * MSUN * uM**-1
Rss = np.array([1.0]) * RSUN * uL**-1
Teffs = np.array([5700])
Mps = np.array([1.0]) * PLANETS.Jupiter.M * uM**-1
Rps = np.array([np.nan]) * PLANETS.Jupiter.R * uL**-1
Per = np.array([50]) * DAY * uT**-1

ams_list = []
nms_list = []
ems_list = []
Tms_list = []
Ems_list = []
ops_list = []
rt_times = []
rt_pers = []
rt_mass = []
roche_lims = []

for Ms, Rs, Teff, Mp, Rp, Porb in zip(Mss, Rss, Teffs, Mps, Rps, Per):

    if pd.isnull(Rp):
        Rp, _, _ = forecaster.Mstat2R(mean=Mp / PLANETS.Jupiter.M * uM, std=0.1, unit='Jupiter', sample_size=200, classify='Yes')
        Rp = Rp * PLANETS.Jupiter.R / uL
    else:
        pass

    # Star properties
    st_life = stellar_lifespan(Ms * uM)

    # Planet properties
    ap = semiMajorAxis(Porb, Ms, Mp)
    npini = meanMotion(ap, Ms, Mp)

    # Roche limit of the moon
    am_roche = aRoche_solid(Mp, Mm, Rm)
    roche_lims.append(am_roche)

    # Initial conditions moon
    amini = 10. * am_roche
    nmini = meanMotion(amini, Mp, Mm)
    pm_ini = 2 * np.pi / nmini
    Tmini = equil_temp(Teff, Rs, ap, Albem)
    Emini = e_tidal(Tmini, nmini, densm=densm, Mm=Mm, Rm=Rm, eccm=emini)

    # Dependent properties
    epsilon = opini / omegaCritic(Mp, Rp)
    k2q = k2Q(alpha0, beta0, epsilon)

    print(Porb / DAY * uT, "days", pm_ini / DAY * uT, "days", Ms / MSUN * uM, "Msun", Mp / PLANETS.Jupiter.M * uM, "Mjup", Rp / PLANETS.Jupiter.R * uL, "Rjup")
    # ************************************************************
    # DERIVED PARAMETERS
    # ************************************************************

    # Arguments
    args = dict(
        qk2q=0, k2q=k2q,
        qRp=0, Rp=Rp,
    )

    # Parameters dictionary
    parameters = dict(Ms=Ms, Mp=Mp, alpha=alpha0, beta=beta0, E_act=E_act, B=B, Ts=Ts, Tb=Tb, T1=T1,
                      Cp=Cp, ktherm=ktherm, Rac=Rac, a2=a2, alpha_exp=alpha_exp, densm=densm, Mm=Mm,
                      Rm=Rm, melt_fr=melt_fr, Rg=Rg, sigmasb=sigmasb, shear_Conv=shear_Conv,
                      visc_Conv=visc_Conv, args=args)

    # ************************************************************
    # INITIAL CONDITIONS FOR THE SYSTEM
    # ************************************************************
    parameters["nm_ini"] = nmini  # Moon's initial mean motion
    parameters["np_ini"] = npini  # PLanet's initial mean motion
    parameters["op_ini"] = opini  # Planet's initial rotation rate
    parameters["Tm_ini"] = Tmini  # Moon's initial temperature
    parameters["Em_ini"] = Emini  # Moon's initial tidal heat
    parameters["em_ini"] = emini  # Moon's initial eccentricity

    # ************************************************************
    # INTEGRATION OF NM+OM
    # ************************************************************
    # t_scale = GYEAR / uT
    t_scale = st_life / uT
    t = 1 * t_scale

    N_steps = 1e5

    # dt_scale = KYEAR / uT
    dt = t / N_steps

    start_time = time.time()
    sys.stdout.write(f"Running simulation for {N_steps} steps...")
    sys.stdout.flush()

    motion_m = Variable('mean_motion_m', nmini)
    omega_p = Variable('omega_planet', opini)
    motion_p = Variable('mean_motion_p', npini)
    temper_m = Variable('temperature', Tmini)
    tidal_m = Variable("tidal_heat", Emini)
    eccen_m = Variable("eccentricity", emini)

    variables = [motion_m, omega_p, motion_p, temper_m, tidal_m, eccen_m]
    if parameters["em_ini"] == 0.0:
        variables = [motion_m, omega_p, motion_p, temper_m, tidal_m]

    simulation = Simulation(variables)

    simulation.set_integration_method("rk4")

    simulation.set_diff_eq(global_differential_equation, **parameters)

    simulation.run(t, dt)

    end_time = time.time()
    exec_time = (end_time - start_time) / MIN
    print(f"\n***********************\n* finished in {exec_time:.3f}m *\n***********************\n")
    # ************************************************************
    # INTEGRATION
    # ************************************************************

    # ************************************************************
    # GET SOLUTIONS
    # ************************************************************
    times, solutions = simulation.history
    nms = solutions[:, 0]
    ops = solutions[:, 1]
    nps = solutions[:, 2]
    Tms = solutions[:, 3]
    Ems = solutions[:, 4]
    if parameters["em_ini"] != 0.0:
        ems = solutions[:, 5]
        ems_list.append(ems)

    # FINDING MOON POSITION
    ams = mean2axis(nms, Mp, Mm)
    ams_list.append(ams)
    nms_list.append(nms)
    Tms_list.append(Tms)
    Ems_list.append(Ems)

    # FILLING OMEGA EVOLUTION
    ops_list.append(ops)

    # ************************************************************
    # FINDING ROUND-TRIP TIMES OR RUNAWAY TIMES
    # ************************************************************
    if args["qk2q"] == 0 and args["qRp"] == 0:
        try:
            pos = np.where(ams <= am_roche)[0][0]
            rt_times += [times[pos] / GYEAR * uT]
            rt_pers += [np.log10(Porb / DAY * uT)]
            rt_mass += [Ms / MSUN * uM]
        except IndexError:
            continue
            # rt_times += [np.nan]

    else:
        try:
            ap = semiMajorAxis(Porb, Ms)
            r_hill = ap * (Mp / (3.0 * Ms))**(1.0 / 3.0)

            pos = np.where(ams[i] >= 0.48 * r_hill[i])[0][0]
            time += [times[pos] / GYEAR]
        except IndexError:
            time += [t / GYEAR]

ams_list = np.array(ams_list)
nms_list = np.array(nms_list)
ops_list = np.array(ops_list)
Tms_list = np.array(Tms_list)
Ems_list = np.array(Ems_list)

if emini != 0.0:
    ems_list = np.array(ems_list)

roche_lims = np.array(roche_lims)


# ************************************************************
# FINDING CHANGE IN RADIUS FOR 1.0 AU
# ************************************************************
# rads = Mp2Rp(Mp, times, **args)
# rads = np.array(rads)

# The heat function for a constant omega (om=oini) with Mathis eq.
# om = 2 * np.pi / Protini
# # beta=alpha2beta(Mp,alpha)
# epsilon = oms_list / omegaCritic(Mp, rads)
# alpha = alpha0 * (PLANETS.Saturn.R / rads)
# # alpha=alpha0
# k2q = k2Q(alpha, beta, epsilon)

rt_times = np.array(rt_times)
rt_pers = np.array(rt_pers)
rt_mass = np.array(rt_mass)

labels = {"P": r"$\mathrm{Log_{10}}(P_\mathrm{orb})\mathrm{[d]}$",
          "Ms": r"$M_\bigstar[\mathrm{M_\odot}]$", "Mp": r"$M_\mathrm{p}[\mathrm{M_{jup}}]$",
          "t": r"$\mathrm{Log_{10}}(\mathrm{Time})\mathrm{[Gyr]}$"}

labelsize = 7.
markersize = 7.
ticksize = 6.

x = times / GYEAR * uT
y = ams_list[0] / roche_lims
z = Tms_list[0]

# dEdts = []
# for i in range(len(Tms_list[0])):
#     dEdts.append(e_tidal(Tms_list[0][i], nms_list[0][i], densm=densm, Mm=Mm, Rm=Rm, Mp=Mps[0],
#                          eccm=ems_list[0][i], shear_Conv=shear_Conv, visc_Conv=visc_Conv))

# dEdts = np.array(dEdts)
# z = surf_temp(dEdts, Rm, sigmasb=sigmasb)


def colorline(x, y, z=None, cmap='copper', linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, linewidth=linewidth, alpha=alpha,
                              norm=plt.Normalize(np.nanmin(z), np.nanmax(z)))

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


# Remove the middle 40% of the RdBu_r colormap
interval = np.hstack([np.linspace(0, 0.45), np.linspace(0.48, 1)])
colors = plt.cm.RdBu_r(interval)
cmap = LinearSegmentedColormap.from_list('name', colors)


fig = plt.figure(figsize=(5, 3.5))
ax = fig.add_subplot(1, 1, 1)
lc = colorline(x, y, z, cmap="jet")
cbar = fig.colorbar(lc, orientation="vertical", aspect=17, format="%.2f", pad=0.04)
cbar.set_label(label="Temperature [K]", size=7)
cbar.set_ticks(np.linspace(np.nanmin(z), np.nanmax(z), 9))
cbar.ax.tick_params(labelsize=ticksize)
ax.axhline(am_roche / roche_lims, c="k", ls="--", lw=0.9, zorder=0.0, label="Roche limit")

# ax.set_xlim(-0.005, np.nanmax(rt_times) + np.nanmax(rt_times) / 20.)
# ax.set_xlim(0.0, 10)
ax.set_ylim(0.0, np.nanmax(y) + 1)
ax.tick_params(axis='both', direction="in", labelsize=ticksize)

ax.set_xlabel("Time [Gyr]", fontsize=labelsize)
ax.set_ylabel(r"Moon semi-major axis [$a_\mathrm{Roche}$]", fontsize=labelsize)
ax.legend(loc="upper right", fontsize=labelsize)
fig.savefig("./Figures/migration.png", dpi=300)

cut = -100
plt.figure(figsize=(6, 4))
plt.plot(times[:cut] / GYEAR * uT, Tms_list[0][:cut], c="k", ls="-")
plt.xlabel("Time [Gyr]")
plt.ylabel("Moon Temperature [K]")
# plt.ylim((0.0, Tmini + 10.0))
plt.savefig("./Figures/temper.png")

if emini != 0.0:
    plt.figure(figsize=(6, 4))
    plt.plot(times[:cut] / GYEAR * uT, ems_list[0][:cut], c="k", ls="-")
    plt.xlabel("Time [Gyr]")
    plt.ylabel("Moon eccentricity")
    # plt.ylim((0.0, emini + 0.005))
    plt.savefig("./Figures/eccentricity.png")


plt.figure(figsize=(6, 4))
plt.plot(Tms_list[0][:cut], Ems_list[0][:cut] * uM * uL**2. * uT**-3., c="k", ls="-")
plt.xlabel("Temperature [K]")
plt.ylabel("Moon Tidal Heat Rate [W]")
plt.savefig("./Figures/heat_tempe.png")


# fig = plt.figure(figsize=(7.0, 5.2))
# # fig.subplots_adjust(wspace=0)
# ax = fig.add_subplot(1, 1, 1, projection="polar")

# colors = ["k", "b", "r", "g"]
# # pol_vec = np.linspace(0, rt_times, 1000000)
# # for i in range(len(apps)):

# sca = ax.scatter(x[::freq], y[::freq], marker='o', c=z[::freq], edgecolor='none', s=25,
#                  linewidths=0.5, cmap=mpl.cm.get_cmap("jet"), zorder=3)
# ax.scatter(0, 0, marker="o", c="k", edgecolor='none', s=40, zorder=5)
# # plt.legend(bbox_to_anchor=(-0.1, 0.9, 1.0, .508),
# #            loc=3, ncol=1, mode="expand", borderaxespad=0., fontsize=14, frameon=False)
# # rlab = plt.ylabel("$a_{\mathrm{p}}\;[\mathrm{au}]$", fontsize=14, labelpad=-142)
# # rlab.set_position((-0.2, 0.59))
# # rlab.set_rotation(-50)
# ax.set_rlabel_position(-35)
# ax.tick_params(axis="both", which="major", labelsize=labelsize)

# label_position = ax.get_rlabel_position()
# # ax.text(np.radians(label_position + 0.65 * GYEAR), ax.get_rmax() / 2.0,
# #         r"$a_{\mathrm{p}}\;[\mathrm{au}]$", rotation=label_position - 15,
# #         ha="center", va="center", fontsize=9)

# # Change units of polar graph
# t = ax.get_xticks()
# ax.set_xticklabels(t, fontsize=labelsize)
# ax.xaxis.set_major_formatter(plticker.FormatStrFormatter("%.1f"))
# # loc = plticker.MultipleLocator(base=0.01)  # this locator puts ticks at regular intervals
# # ax.yaxis.set_major_locator(loc)
# # ax.fill_between(pol_vec, 0.02, 0.04, color="lightgray")
# ax.plot(times, np.ones(len(times)), ls="-.", lw=1.7)
# ax.set_xlabel("Time [Gyr]", fontsize=14)
# # ax.set_title(r"$Semimajor\;Axis$", fontsize=13, x=0.5, y=-0.315,  # x=0.1, y=-0.115
# #              color='white', ha="center", va="center",
# #              bbox=dict(alpha=1.0, facecolor='gray', edgecolor='none', boxstyle='round'))
# ax.set_theta_direction(-1)
# ax.set_theta_offset(np.pi / 2.0)
# ax.set_xlim((0.0, 0.8))

# cbar = fig.colorbar(sca, orientation="vertical", shrink=0.9, aspect=17, format="%.1f", pad=0.08)
# cbar.set_label(label="Temperature [K]", size=7)
# # cbar.ax.tick_params(labelsize=6, rotation=25)
# # cbar.set_ticks(np.linspace(np.min(Tms_list[0]), np.max(Tms_list[0]), 9))

# fig.tight_layout()
# fig.savefig("Figures/migration_polar.png", dpi=300)

exit(0)

fig = plt.figure(figsize=(3, 3))

ax = fig.add_subplot(1, 1, 1)

x_min = np.min(rt_mass)
x_max = np.max(rt_mass)

Mss_vector = np.linspace(x_min - 0.001, x_max + 0.001, 100)
lifespan = np.log10(stellar_lifespan(Mss_vector * MSUN) / GYEAR)

sc_0 = ax.scatter(rt_mass, np.log10(rt_times), marker='o', c=rt_pers, edgecolor='k', s=markersize,
                  linewidths=0.5, cmap=plt.cm.gnuplot, zorder=3, label="Moon migration")

ax.plot(Mss_vector, lifespan, c="k", ls="--", lw=0.5, label="Stellar lifespan")
ax.set_xlim((x_min - 0.01, x_max + 0.01))
ax.set_ylabel(labels["t"], fontsize=labelsize)
ax.set_xlabel(labels["Ms"], fontsize=labelsize)
ax.tick_params(axis='both', direction="in", labelsize=ticksize)
ax.grid(alpha=0.4, ls="--")
ax.legend(loc="upper center", fontsize=5, ncol=2, framealpha=0.6)
# ax1 = ax.twinx()
# Mss_vector = np.linspace(x_min, x_max, 100)
# ax1.plot(Mss_vector, stellar_lifespan(Mss_vector * MSUN) / GYEAR, c="b", ls="--", lw=0.5)
# ax1.set_ylabel("Stellar lifespan [Gyr]", fontsize=labelsize, color="b")
# # ax1.set_xlabel("time [Myr]", fontsize=14)
# ax1.set_xlim((x_min, x_max))
# lim1 = ax.get_ylim()
# lim2 = ax1.get_ylim()
# f = lambda x: lim2[0] + (x - lim1[0]) / (lim1[1] - lim1[0]) * (lim2[1] - lim2[0])
# ticks = f(ax.get_yticks())
# ax1.yaxis.set_major_locator(plticker.FixedLocator(ticks))
# # ax1.yaxis.set_major_formatter(plticker.FuncFormatter(fmt))
# ax1.tick_params(axis="x", which="major", labelsize=15)
# ax1.tick_params(axis="y", which="major", colors="b", labelsize=7)

cbar = fig.colorbar(sc_0, orientation="horizontal", shrink=0.9, aspect=17, format="%.3f")
cbar.set_label(label=labels["P"], size=labelsize)
cbar.ax.tick_params(labelsize=ticksize, rotation=25)
cbar.set_ticks(np.linspace(np.min(rt_pers), np.max(rt_pers), 9))

fig.tight_layout()
fig.savefig("./Figures/scatter.png", dpi=300)

exit(0)
