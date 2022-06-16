.. _quickstart:

Please visit our quickstart guide at:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    
    from ploonetide import TidalSimulation
    from ploonetide.utils import colorline
    from ploonetide.utils.functions import mean2axis, find_moon_fate
    from ploonetide.utils.constants import GYEAR, DAY, MSUN, AU
    
    # ************************************************************
    # INTEGRATION
    # ************************************************************
    simulation = TidalSimulation(
        system='planet-moon',
        planet_orbperiod=20,
        moon_eccentricty=0.0,
        moon_semimaxis=10,
        planet_size_evolution=False,
        planet_internal_evolution=False,
        planet_core_dissipation=False,
    )
    
    
    integration_time = 1 * simulation.stellar_lifespan
    N_steps = 1e5
    timestep = integration_time / N_steps

    simulation.set_integration_method('rk4')
    simulation.set_diff_eq()
    simulation.run(integration_time, timestep)