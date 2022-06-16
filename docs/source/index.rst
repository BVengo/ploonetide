.. title:: Ploonetide docs

.. rst-class:: frontpage

**********
Ploonetide
**********

**Ploonetide is a friendly-user package for calculating tidal evolution of compact systems with Python.**

|test-badge| |pypi-badge| |pypi-downloads| |astropy-badge| |docs-badge|

.. |pypi-badge| image:: https://badge.fury.io/py/ploonetide.svg
                :target: https://badge.fury.io/py/ploonetide
.. |pypi-downloads| image:: https://pepy.tech/badge/ploonetide/month
                :target: https://pepy.tech/project/ploonetide
.. |test-badge| image:: https://github.com/JAAlvarado-Montes/ploonetide/workflows/ploonetide-build-test/badge.svg
                 :target: https://github.com/JAAlvarado-Montes/ploonetide/actions?query=workflow%3Aploonetide-build-test
.. |astropy-badge| image:: https://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
                   :target: http://www.astropy.org
.. |docs-badge| image:: https://readthedocs.org/projects/ploonetide/badge/?version=latest
                 :target: https://ploonetide.readthedocs.io/en/latest/?badge=latest
                 :alt: Documentation Status


**Ploonetide** is an open-source Python package which offers a simple and user-friendly way
to calculate tidal evolution of compact planetary systems.

.. Image:: ./_static/images/logo.png

Documentation
-------------

Read the documentation at `https:// <https://>`_.


Quickstart
----------

Please visit our quickstart guide at `https:// <https://>`_.

´´´

>>> import numpy as np
>>> import matplotlib.pyplot as plt

>>> from ploonetide import TidalSimulation
>>> from ploonetide.utils import colorline
>>> from ploonetide.utils.functions import mean2axis, find_moon_fate
>>> from ploonetide.utils.constants import GYEAR, DAY, MSUN, AU

>>> # ************************************************************
>>> # INTEGRATION
>>> # ************************************************************
>>> simulation = TidalSimulation(
>>>     system='planet-moon',
>>>     planet_orbperiod=20,
>>>     moon_eccentricty=0.0,
>>>     moon_semimaxis=10,
>>>     planet_size_evolution=False,
>>>     planet_internal_evolution=False,
    planet_core_dissipation=False,
>>> )


>>> integration_time = 1 * simulation.stellar_lifespan
>>> N_steps = 1e5
>>> timestep = integration_time / N_steps

>>> simulation.set_integration_method('rk4')
>>> simulation.set_diff_eq()
>>> simulation.run(integration_time, timestep)
´´´


Contributing
------------

We welcome community contributions!
Please read the  guidelines at `https:// <https://>`_.


Citing
------

If you find Ploonetide useful in your research, please cite it and give us a GitHub star!
Please read the citation instructions at `https:// <https://>`_.


Contact
-------
Ploonetide is an open source community project created by `the authors <AUTHORS.rst>`_.
The best way to contact us is to `open an issue <https://github.com/JAAlvarado-Montes/ploonetide/issues/new>`_ or to e-mail  jaime-andres.alvarado-montes@hdr.mq.edu.au.

.. toctree::
    .. autosummary::
       :toctree: generated
    :maxdepth: 3

    reference_guide

    What's new? <whats-new-v2.ipynb>
    quickstart
    tutorials/index
    reference/index
    about/index