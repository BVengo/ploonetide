.. _installation:

Installation
============

Create a virtual environment
----------------------------

ploonetide works with `Python 3.6 or above`_. It is recommended you create a dedicated `Python environment`_ before you install spike2py. In your project directory, run the following commands:

.. code-block:: bash

   python -m venv env

Then activate your new virtual environment.

On macOS and Linux:

.. code-block:: bash

   source env/bin/activate

On Windows:

.. code-block:: bash

   .\env\Scripts\activate

Install spike2py and its dependencies
-------------------------------------

With your virtual environment activated, run the following command:

.. code-block:: bash

   pip install ploonetide

Testing your ploonetide installation
----------------------------------

With your virtual environment activated, start Python and type the following:

.. code-block:: python

    import ploonetide
    simulation = ploonetide.TidalSimulation()

You should see a print statement.


.. _Python 3.6 or above: https://www.python.org/downloads/
.. _Python environment: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment