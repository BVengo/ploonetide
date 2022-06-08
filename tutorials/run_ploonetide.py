import numpy as np
import matplotlib.pyplot as plt

from ploonetide import TidalSimulation
from ploonetide.utils import colorline
from ploonetide.utils.functions import mean2axis, semiMajorAxis
from ploonetide.utils.constants import GYEAR, DAY, MSUN, AU

# ************************************************************
# INTEGRATION
# ************************************************************
simulation = TidalSimulation(system='planet-moon',
                             planet_orbperiod=100,
                             moon_eccentricty=0.3,
                             moon_semimaxis=20,
                             planet_size_evolution=False,
                             planet_internal_evolution=False,
                             planet_core_dissipation=False,
                             star_internal_evolution=False)

integration_time = 1 * simulation.stellar_lifespan
N_steps = 1e4
timestep = integration_time / N_steps

simulation.set_integration_method('rk4')
simulation.set_diff_eq()
simulation.run(integration_time, timestep)

# ************************************************************
# GET SOLUTIONS
# ************************************************************
times, solutions = simulation.history
nms = solutions[:, 0]
ops = solutions[:, 1]
nps = solutions[:, 2]
Tms = solutions[:, 3]
Ems = solutions[:, 4]
if simulation.moon_eccentricty != 0.0:
    ems = solutions[:, 5]

# COMPUTE SEMIMAJOR AXIS
ams = mean2axis(nms, simulation.planet_mass, simulation.moon_mass)
aps = mean2axis(nps, simulation.star_mass, simulation.planet_mass)
# ams_list.append(ams)
# nms_list.append(nms)
# Tms_list.append(Tms)
# Ems_list.append(Ems)

# # FILLING OMEGA EVOLUTION
# # ops_list.append(ops)

# ************************************************************
# FINDING ROUND-TRIP TIMES OR RUNAWAY TIMES
# ************************************************************
try:
    pos = np.where(ams <= simulation.moon_roche_radius)[0][0]
    rt_times = [times[pos] / GYEAR]
    rt_pers = [np.log10(simulation.planet_orbperiod / DAY)]
    rt_mass = [simulation.star_mass / MSUN]

    rt_times = np.array(rt_times)
    rt_pers = np.array(rt_pers)
    rt_mass = np.array(rt_mass)
except IndexError:
    try:
        ap = semiMajorAxis(simulation.planet_orbperiod, simulation.star_mass, simulation.planet_mass)
        r_hill = ap * (simulation.planet_mass / (3.0 * simulation.star_mass))**(1.0 / 3.0)
        pos = np.where(ams >= 0.48 * r_hill)[0][0]
        rt_times = [times[pos] / GYEAR]
    except IndexError:
        rt_times = [integration_time / GYEAR]

print(rt_times)


ams_list = np.array(ams)
aps_list = np.array(aps)
nms_list = np.array(nms)
nps_list = np.array(nps)
ops_list = np.array(ops)
Tms_list = np.array(Tms)
Ems_list = np.array(Ems)

# if simulation.moon_eccentricty != 0.0:
#     ems_list = np.array(ems)

# roche_lims = np.array(roche_lims)

# nps = solutions[:, 0]
# oms = solutions[:, 1]
# eps = solutions[:, 2]
# osms = solutions[:, 3]
# mps = solutions[:, 4]
# aps = mean2axis(nps, simulation.star_mass, mps)

# fig = plt.figure(figsize=(5, 3.5))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(times / GYEAR, aps / AU, 'k-')
# fig.savefig("migration_star-planet.png", dpi=300)

# exit(0)

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

labels = {"P": r"$\mathrm{Log_{10}}(P_\mathrm{orb})\mathrm{[d]}$",
          "Ms": r"$M_\bigstar[\mathrm{M_\odot}]$", "Mp": r"$M_\mathrm{p}[\mathrm{M_{jup}}]$",
          "t": r"$\mathrm{Log_{10}}(\mathrm{Time})\mathrm{[Gyr]}$"}

labelsize = 7.
markersize = 7.
ticksize = 6.

x = times / GYEAR
y = ams_list / simulation.moon_roche_radius
z = Tms_list


# fig = plt.figure(figsize=(5, 3.5))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(times / GYEAR, aps / AU, 'k-')
# fig.savefig("migration_planet.png", dpi=300)

# exit(0)

# dEdts = []
# for i in range(len(Tms_list[0])):
#     dEdts.append(e_tidal(Tms_list[0][i], nms_list[0][i], densm=densm, Mm=Mm, Rm=Rm, Mp=Mps[0],
#                          eccm=ems_list[0][i], shear_Conv=shear_Conv, visc_Conv=visc_Conv))

# dEdts = np.array(dEdts)
# z = surf_temp(dEdts, Rm, sigmasb=sigmasb)


fig = plt.figure(figsize=(5, 3.5))
ax = fig.add_subplot(1, 1, 1)
lc = colorline(x, y, z, cmap="jet")
cbar = fig.colorbar(lc, orientation="vertical", aspect=17, format="%.2f", pad=0.04)
cbar.set_label(label="Temperature [K]", size=7)
cbar.set_ticks(np.linspace(np.nanmin(z), np.nanmax(z), 9))
cbar.ax.tick_params(labelsize=ticksize)
# ax.axhline(am_roche / roche_lims, c="k", ls="--", lw=0.9, zorder=0.0, label="Roche limit")

# ax.set_xlim(-0.005, np.nanmax(rt_times) + np.nanmax(rt_times) / 20.)
# ax.set_xlim(0.0, 10)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(0.0, np.nanmax(y) + 1)
ax.tick_params(axis='both', direction="in", labelsize=ticksize)

ax.set_xlabel("Time [Gyr]", fontsize=labelsize)
ax.set_ylabel(r"Moon semi-major axis [$a_\mathrm{Roche}$]", fontsize=labelsize)
ax.legend(loc="upper right", fontsize=labelsize)
fig.savefig("migration.png", dpi=300)
