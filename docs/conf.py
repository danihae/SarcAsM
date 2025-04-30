# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------
import os
import sys
import subprocess

# Add base of package to path
sys.path.insert(0, os.path.abspath('..'))
from sarcasm import __version__ as package_version

# -- Project information -----------------------------------------------------
project = 'SarcAsM'
copyright = '2025, University Medical Center Göttingen'
author = 'Daniel Haertter'
release = package_version
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'autoapi.extension',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.autosummary',
    'nbsphinx'
]

autoapi_type = 'python'
autoapi_dirs = ['../sarcasm', '../contraction_net']
autoapi_ignore = [
    "*/test*.py", "*/tests/*.py", "*/type_utils.py",
    "*/siam_unet/*", "*/progress/*"
]

autosummary_generate = True
nbsphinx_execute = 'never'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']  # Standard Sphinx convention

def run_feature_dict_script():
    script_path = os.path.abspath('./feature_dict.py')
    subprocess.run([sys.executable, script_path], check=True)

def setup(app):
    app.connect('builder-inited', lambda app: run_feature_dict_script())

source_suffix = '.rst'
master_doc = 'index'
language = 'en'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
pygments_style = None

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'logo_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
}
html_logo = 'images/logo.png'
html_static_path = ['_static']  # Standard, even if empty

# GitHub integration for "Edit on GitHub" links
html_context = {
    'display_github': True,
    'github_user': 'danihae',
    'github_repo': 'SarcAsM',
    'github_version': 'main',      # Use 'main' or 'master' as appropriate
    'conf_py_path': '/docs/',      # With leading and trailing slash
}

# -- Options for HTMLHelp output ---------------------------------------------
htmlhelp_basename = 'sarcasmdoc'

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}
latex_documents = [
    (master_doc, 'sarcasm_docs.tex', 'SarcAsM Documentation',
     'Daniel Haertter', 'manual'),
]
latex_logo = 'images/logo.png'
latex_engine = 'lualatex'

# -- Options for manual page output ------------------------------------------
man_pages = [
    (master_doc, 'sarcasm_docs', 'SarcAsM Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (master_doc, 'sarcasm_docs', 'SarcAsM Documentation',
     author, 'SarcAsM', 'Sarcomere Analysis Multitool',
     'Miscellaneous'),
]

# -- Options for Epub output -------------------------------------------------
epub_title = project
epub_exclude_files = ['search.html']

# -- Extension configuration -------------------------------------------------
todo_include_todos = False
