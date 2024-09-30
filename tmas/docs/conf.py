# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------

project = u"tmas"
copyright = u"2024, RMIT 2OO2 TEAM"
author = u"RMIT 2OO2 TEAM"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

import os
import sys
source_suffix = ['.rst', '.md']
sys.path.insert(0, os.path.abspath('../tmas'))

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # For Google-style and NumPy-style docstrings
    "sphinx.ext.viewcode",  # To link source code
    "sphinx_autodoc_typehints",
    "autoapi.extension",
]
autoapi_dirs = ["../"]
autoapi_options = ['members', 'undoc-members']


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

