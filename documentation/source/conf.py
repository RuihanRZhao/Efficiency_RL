# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os, sys

project = 'Efficiency_RL'
copyright = '2023, Ruihan Zhao'
author = 'Ruihan Zhao'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []
nitpicky = []
templates_path = ['_templates']
exclude_patterns = []


# The master toctree document.
root_doc = 'index'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']


sys.path.insert(0, os.path.abspath(os.path.join('..', '..', 'src')))


extensions.append('sphinx.ext.autodoc')
extensions.append('sphinx.ext.viewcode')
