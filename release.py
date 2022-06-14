#!/usr/bin/env python
import os
import sys

# Prepare and send a new release to PyPI
os.system('python -m pip install build twine')
os.system('python -m build')
os.system('twine check dist/*')
os.system('twine upload -r testpypi dist/*')
os.system('twine upload dist/*')
os.system('rm -rf dist/ploonetide*')
sys.exit()
