# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

# import os
# import sys
# sys.path.insert(0, os.path.abspath('../VortexAD/core'))     # for autodoc

# -- Project information -----------------------------------------------------

project = 'VortexAD'
copyright = '2025, Luca Scotzniovsky'
author = 'Luca Scotzniovsky'
version = '0.0.0'
# release = 0.1.0rtc


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "autoapi.extension",
    "numpydoc",                 
    "sphinx_copybutton",            # allows copying code embedded in the docs rendered from .md or .ipynb files
    "myst_nb",                      # renders .md, .myst, .ipynb files
    "sphinx.ext.viewcode",          # adds the source code for classes and functions in auto generated api ref
    # "sphinxcontrib.collections",    # adds files from outside src and executes functions before Sphinx builds
    # "sphinx_collections",           # adds files from outside src and executes functions before Sphinx builds
    "sphinxcontrib.bibtex",         # for references and citations
]

# import sphinx as aa
# print(aa.__version__)

# from pip import _internal
# _internal.main(['list'])

# sphinxcontrib.bibtex options
bibtex_bibfiles = ['src/references.bib']

# myst_nb options
myst_title_to_header = True
myst_enable_extensions = ["dollarmath", "amsmath", "tasklist"]
nb_execution_mode = 'off'

# autoapi options
autoapi_dirs = ["../VortexAD/core"]
autoapi_root = 'src/autoapi'
autoapi_type = 'python'
autoapi_file_patterns = ['*.py', '*.pyi']
autoapi_options = [ 'members', 'undoc-members', 'private-members', 'show-inheritance', 
                   'show-module-summary', 'special-members', 'imported-members', ]
autoapi_add_toctree_entry = False
autoapi_member_order = 'groupwise'
autoapi_python_class_content = 'class' # 'both' or '__init'

root_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['README.md', '_build', 'Thumbs.db', '.DS_Store', 'src/welcome.md']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme' # other theme options: 'sphinx_book_theme', 'sphinx_rtd_theme', 
                                # 'alabaster', 'classic', 'sphinxdoc', 'nature', 'bizstyle', ...

# html_theme_options for sphinx_rtd_theme
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',   # other valid colors: 'white', ...
    # toc options
    'collapse_navigation': False,   # default: True
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': True     # default: False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


import os
import re
import shutil
from pathlib import Path


_DOCS = Path(__file__).resolve().parent
_REPO = _DOCS.parent
_TEMP = _DOCS / "src" / "_temp"


def _py2md(path):
    """Convert an example Python file to a Markdown documentation page."""
    code = path.read_text(encoding="utf-8")

    single_start = code.find("'''")
    double_start = code.find('"""')

    if single_start == -1 and double_start == -1:
        raise SyntaxError(
            f"{path}: docstring for title and description is not declared"
        )

    if single_start == -1:
        quotes = '"'
    elif double_start == -1:
        quotes = "'"
    elif single_start < double_start:
        quotes = "'"
    else:
        quotes = '"'

    pattern = r"'''(.*?)'''" if quotes == "'" else r'"""(.*?)"""'
    match = re.search(pattern, code, re.DOTALL)

    if match is None:
        raise SyntaxError(
            f"{path}: docstring for title and description is not declared correctly"
        )

    docstring = match.group(1).strip()
    parts = docstring.split(":", 1)

    title = parts[0].strip()
    description = parts[1].strip() if len(parts) == 2 else ""

    output = (
        f"# {title}\n\n"
        f"{description}\n\n"
        "```python\n"
        f"{code}\n"
        "```\n"
    )

    path.with_suffix(".md").write_text(output, encoding="utf-8")


def _stage_examples_and_tutorials(app, config):
    """Prepare examples and tutorials for the Sphinx build."""
    shutil.rmtree(_TEMP, ignore_errors=True)

    for name in ("tutorials", "examples"):
        source = _REPO / name
        destination = _TEMP / name

        shutil.copytree(source, destination)

    for example in (_TEMP / "examples").glob("**/ex_*.py"):
        _py2md(example)


def setup(app):
    app.connect("config-inited", _stage_examples_and_tutorials)

collections_target = 'src/_temp'    # default : '_collections', the default storage location for all collections
collections_clean  = True           # default : True, all configured target locations get wiped out at the beginning
                                    # can be overwritten for individual collection by setting value for the 'clean' key
collections_final_clean  = True     # default : True, all collections start their clean-up routine after a Sphinx build is done
                                    # can be overwritten for individual collection by setting value for the 'final_clean' key
