#!/usr/bin/env python
import os
import sys
import setuptools

from distutils.util import convert_path

# Prepare and send a new release to PyPI
if 'release' in sys.argv[-1]:
    os.system('python setup.py sdist')
    os.system('pip install twine bumpver')
    os.system('twine check dist/*')
    os.system('twine upload -r testpypi dist/*')
    os.system('twine upload dist/*')
    os.system('rm -rf dist/ploonetide*')
    sys.exit()

# DEPENDENCIES
# 1. What are the required dependencies?
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()
# 2. What dependencies required to run the unit tests? (i.e. `pytest --remote-data`)
tests_require = ['pytest', 'pytest-cov', 'pytest-remotedata', 'codecov',
                 'pytest-doctestplus', 'codacy-coverage']
# 3. What dependencies are required for optional features?
# `BoxLeastSquaresPeriodogram` requires astropy>=3.1.
# `interact()` requires bokeh>=1.0, ipython.
# `PLDCorrector` requires pybind11, celerite.
extras_require = {"all": ["astropy>=3.1",
                          "bokeh>=1.0", "ipython",
                          "pybind11", "celerite"],
                  "test": tests_require}

# Load the __version__ variable without importing the package already
main_ns = {}
ver_path = convert_path('src/ploonetide/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setuptools.setup(
    name='ploonetide',
    version=main_ns['__version__'],
    description="Calculate tidal interactions in planetary systems",
    long_description=open('README.rst').read(),
    long_description_content_type='text/markdown',
    author='Jaime Andrés Alvarado Montes',
    author_email='jaime-andres.alvarado-montes@hdr.mq.edu.au',
    url='https://github.com/JAAlvarado-Montes/ploonetide',
    license='MIT',
    package_dir={'': 'src/ploonetide'},
    packages=setuptools.find_packages(where='src/ploonetide/'),
    python_requires='>=3.6',
    install_requires=install_requires,
    extras_require=extras_require,
    setup_requires=['pytest-runner'],
    tests_require=tests_require,
    include_package_data=True,
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
