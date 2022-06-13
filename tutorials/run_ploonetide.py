import numpy as np
import matplotlib.pyplot as plt

from ploonetide import TidalSimulation
from ploonetide.utils import colorline
from ploonetide.utils.functions import mean2axis, find_moon_fate
from ploonetide.utils.constants import GYEAR, DAY, MSUN, AU

# ************************************************************
# INTEGRATION
# ************************************************************
simulation = TidalSimulation(system='planet-moon',
                             planet_orbperiod=20,
                             moon_eccentricty=0.0,
                             moon_semimaxis=10,
                             planet_size_evolution=False,
                             planet_internal_evolution=False,
                             planet_core_dissipation=False,
                             star_internal_evolution=False)


integration_time = 1 * simulation.stellar_lifespan
N_steps = 1e5
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

# ************************************************************
# FINDING ROUND-TRIP TIMES OR RUNAWAY TIMES
# ************************************************************
fate = find_moon_fate(times, ams, simulation.moon_roche_radius, simulation.planet_orbperiod,
                      simulation.planet_mass, simulation.star_mass)


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

labels = {'P': r'$\mathrm{Log_{10}}(P_\mathrm{orb})\mathrm{[d]}$',
          'Ms': r'$M_\bigstar[\mathrm{M_\odot}]$', 'Mp': r'$M_\mathrm{p}[\mathrm{M_{jup}}]$',
          't': r'$\mathrm{Log_{10}}(\mathrm{Time})\mathrm{[Gyr]}$'}

labelsize = 7.
markersize = 7.
ticksize = 6.

x = times / GYEAR
y = ams_list / simulation.moon_roche_radius
z = Tms_list

print(y)

# fig = plt.figure(figsize=(5, 3.5))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(times / GYEAR, aps / AU, 'k-')
# fig.savefig("migration_planet.png", dpi=300)

# exit(0)

# dEdts = []
# for i in range(len(Tms_list)):
#     dEdts.append(e_tidal(Tms_list[i], nms_list[i], densm=simulation.moon_density,
#                  Mm=simulation.moon_mass, Rm=simulation.moon_radius, Mp=simulation.planet_mass,
#                  eccm=Ems_list[i]))

# dEdts = np.array(dEdts)
# z = surf_temp(dEdts, simulation.moon_radius)


fig = plt.figure(figsize=(5, 3.5))
ax = fig.add_subplot(1, 1, 1)
lc = colorline(x, y, z, cmap='jet')
cbar = fig.colorbar(lc, orientation='vertical', aspect=17, format="%.2f", pad=0.04)
cbar.set_label(label='Temperature [K]', size=7)
cbar.set_ticks(np.linspace(np.nanmin(z), np.nanmax(z), 9))
cbar.ax.tick_params(labelsize=ticksize)
# ax.axhline(am_roche / roche_lims, c="k", ls="--", lw=0.9, zorder=0.0, label="Roche limit")

ax.set_xlim(x.min(), x[fate.index] + 0.2)
ax.set_ylim(1.0, np.nanmax(y[np.isfinite(y)]) + 1)
ax.tick_params(axis='both', direction='in', labelsize=ticksize)

ax.set_xlabel('Time [Gyr]', fontsize=labelsize)
ax.set_ylabel(r'Moon semi-major axis [$a_\mathrm{Roche}$]', fontsize=labelsize)
# ax.legend(loc='upper right', fontsize=labelsize)
fig.savefig('migration.png', dpi=300)
