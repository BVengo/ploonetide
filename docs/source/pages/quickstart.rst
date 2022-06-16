.. _Quick start:

Quick start
-----------

The code snippet below is how **ploonetide** should be used, for other examples please see the documentation.

.. code-block:: python

    import numpy as np
    
    from ploonetide import TidalSimulation
    
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