import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
#
#  Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'jax-sn'
copyright = '2026, Timo Bogaarts'
author = 'Timo Bogaarts'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
]

templates_path = ['_templates']
exclude_patterns = []

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']



html_css_files = ["css/custom.css"]

autodoc_default_options = {
    "members": True, # documents all members, including dataclasses fields
    "member-order": "bysource", # order members as they appear in the source code
}

typehints_use_signature = True # show typehints in the function signature.
autodoc_preserve_defaults = True # show default values as written in source, not repr()
add_module_names = False # removes the long jax_sn.module.name from the docs. Can re-enable for clarity, but we don't use classes from other modules much, so in principle I don't see too much objections.

rst_prolog = """
.. role:: parallel
   :class: parallel

.. role:: loop
   :class: loop
"""