import os
import sys

from distutils.util import convert_path
sys.path.append(os.path.abspath(
    os.path.join(__file__, "../../src/")
))


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages'
]

autosummary_generate = True

autoclass_content = 'both'

# Exclude build directory and Jupyter backup files:
exclude_patterns = ['_build', '**.ipynb_checkpoints']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

master_doc = 'index'

# Load the __version__ variable without importing the package already
main_ns = {}
ver_path = convert_path('../../src/ploonetide/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

# The full version, including alpha/beta/rc tags.
release = main_ns['__version__']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Execute notebooks? Possible values: 'always', 'never', 'auto' (default)
nbsphinx_execute = "auto"

# Some notebook cells take longer than 60 seconds to execute
nbsphinx_timeout = 300

# General information about the project.
project = f'Ploonetide v{release}'
copyright = 'Jaime A. Alvarado-Montes'
author = 'Ploonetide developers'


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["**/.ipynb_checkpoints"]

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Execute notebooks? Possible values: 'always', 'never', 'auto' (default)
nbsphinx_execute = "auto"

# Some notebook cells take longer than 60 seconds to execute
nbsphinx_timeout = 300

# -- Options for HTML output ----------------------------------------------
# html_theme = 'pydata_sphinx_theme'
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    "external_links": [],
    "github_url": "https://github.com/JAAlvarado-Montes/ploonetide",
}


html_title = "Ploonetide "

html_static_path = ['_static']


# Raw files we want to copy using the sphinxcontrib-rawfiles extension:
# - CNAME tells GitHub the domain name to use for hosting the docs
# - .nojekyll prevents GitHub from hiding the `_static` dir
rawfiles = ['CNAME', '.nojekyll']

# Make sure text marked up `like this` will be interpreted as Python objects
default_role = 'py:obj'

# intersphinx enables links to classes/functions in the packages defined here:
intersphinx_mapping = {'python': ('https://docs.python.org/3/', None),
                       'numpy': ('https://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
                       'matplotlib': ('https://matplotlib.org', None),
                       'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
                       'astropy': ('https://docs.astropy.org/en/latest/', None)}
