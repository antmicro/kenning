# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

"""
Configuration for Kenning Sphinx-based documentation.
"""

import os
import sys

from antmicro_sphinx_utils.defaults import (
    antmicro_html,
    antmicro_latex,
)
from antmicro_sphinx_utils.defaults import (
    extensions as default_extensions,
)
from antmicro_sphinx_utils.defaults import (
    myst_enable_extensions as default_myst_enable_extensions,
)
from antmicro_sphinx_utils.defaults import (
    numfig_format as default_numfig_format,
)

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------

project = "Kenning"
basic_filename = "kenning"
copyright = "2020-2025, Antmicro"
authors = "Antmicro"

# The short X.Y version
version = ""
# The full version, including alpha/beta/rc tags
release = ""

# This is temporary before the clash between myst-parser and immaterial is
# fixed
sphinx_immaterial_override_builtin_admonitions = True

# -- General configuration ---------------------------------------------------
numfig = True
numfig_format = default_numfig_format

# If you need to add extensions just add to those lists
extensions = list(
    set(
        default_extensions
        + [
            "sphinx.ext.githubpages",
            "sphinx.ext.napoleon",
            "sphinx.ext.extlinks",
            # Extension injecting HTML to figures
            "kenning.utils.sphinx_html_as_figure",
            # Extensions enabling save-as feature for directives
            "kenning.utils.sphinx_literalinclude",
            "kenning.utils.sphinx_codeblock",
        ]
    )
)

myst_enable_extensions = default_myst_enable_extensions + ["attrs_block"]
myst_heading_anchors = 3

dev = "https://github.com/antmicro/kenning"

gpu_info_admonition = """
:::{info}
This example requires a CUDA-enabled GPU for proper execution, along with following dependencies:

* CUDA (11.8 is recommended version)
* CUDNN (8 is recommended version)
* NVCC
* NVRTC
* CUDA Toolkit
* NVML (for report generation)
:::
"""  # noqa: E501

extlinks = {"issue": (dev + "issues/%s", "#")}

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "1.8"

todo_include_todos = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The encoding of source files.
# source_encoding = 'utf-8-sig'

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

language = "en"

# A list of paths that contain extra files not directly related to the
# documentation, such as robots.txt or .htaccess. Relative paths are taken as
# relative to the configuration directory. They are copied to the output
# directory. They will overwrite any existing file of the same name.
html_extra_path = ["generated"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["generated/*.rst", "generated/*.md"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

today_fmt = "%Y-%m-%d"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_immaterial"

html_last_updated_fmt = today_fmt

html_show_sphinx = False

(html_logo, html_theme_options, html_context) = antmicro_html(
    gh_slug="antmicro/kenning", pdf_url=f"{basic_filename}.pdf"
)

# The name for this set of Sphinx documents. If None, it defaults to
# "<project> v<release> documentation".
html_title = project

# A shorter title for the navigation bar. Default is the same as html_title.
# html_short_title = None

html_show_sourcelink = False

html_static_path = ["_static"]

html_css_files = [
    "css/bokeh.css",
    "css/compatibility.css",
]

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = basic_filename

# -- Options for LaTeX output ------------------------------------------------

(
    latex_elements,
    latex_documents,
    latex_logo,
    latex_additional_files,
) = antmicro_latex(basic_filename, authors, project)

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]


# -- Napoleon settings -------------------------------------------------------

napoleon_numpy_docstring = True

# -- Extension configuration -------------------------------------------------
rst_epilog = """
.. |project| replace:: %s
.. |projecturl| replace:: `%s <%s>`__
""" % (project, project, dev)

myst_substitutions = {
    "project": project,
    "projecturl": f"[{project}]({dev})",
    "json_compilation_script": "`kenning.scenarios.inference_tester`",
    "json_flow_runner_script": "`kenning.scenarios.json_flow_runner`",
    "optimization_runner_script": "`kenning.scenarios.optimization_runner`",
    "uses_gpu": gpu_info_admonition,
}

linkcheck_anchors_ignore_for_url = ("https:\/\/github\.com.*",)
