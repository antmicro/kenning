# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Sphinx extension for allowing HTML files to be loadable
in figures.
"""
import copy
import logging
from pathlib import Path
from docutils import nodes
from sphinx.util.docutils import SphinxTranslator
from sphinx.application import Sphinx, logger as sphinx_logger
from sphinx.writers.html5 import HTML5Translator


def extend_default_translator(app: Sphinx):
    """
    Function extends HTML translator.

    Function is called when builder is initialized. For HTML format, it extend
    currently used translator with feature for selecting images in HTML format,
    if they're avaiable
    """
    if app.builder.format != 'html':
        return

    translator_class = app.registry.get_translator_class(app.builder)
    if translator_class is None:
        translator_class = HTML5Translator

    class SelectHTMLImageTranslator(translator_class):
        """
        Custom Translator for HTML, with feature for selecting images in HTML
        format if avaiable
        """
        logger = logging.getLogger(__name__)
        BACKUP_IMAGE, BACKUP_RAW = None, None

        def visit_image(self: SphinxTranslator, image: nodes.image):
            """
            Method chooses image in HTML format.

            Method executed when node is visited - right before builder
            (for HTML files) write this node to file in proper format. It swaps
            image with raw html node if there is avaiable image in HTML format.
            It doesn't return anything, but can have influence on output file
            by editing self.body.

            Parameters
            ----------
            image : nodes.image
                Processed node

            Note
            ----
            To preserve original state of nodes (e.g. for other builds), this
            method save state of image node to global variable BACKUP_IMAGE and
            the original state of node is restored during departure - method
            depart_image
            """
            if 'candidates' not in image or len(image['candidates']) == 1:
                return super().visit_image(image)
            if 'image/x-html' not in image['candidates']:
                return super().visit_image(image)
            # Backup image
            self.BACKUP_IMAGE = copy.deepcopy(image)
            # Create raw node from image data
            filepath = image['candidates']['image/x-html']
            with open(Path(self.settings.env.srcdir, filepath), 'r') as fd:
                html = fd.read()
            raw = nodes.raw('', text=html, format='html')
            self.BACKUP_RAW = raw
            # Replacing image with raw
            image.replace_self([raw])
            super().dispatch_visit(raw)

        def depart_image(self, image: nodes.image):
            """
            Method reverts changes made by visit_image.

            Method executed when visit in node ended - right after builder
            (for HTML files) wrote this node to file in proper format. It swaps
            raw html with backuped image node. It doesn't return anything, but
            can have influence on output file by editing self.body.

            Parameters
            ----------
            image : nodes.image
                Processed node
            """
            if self.BACKUP_IMAGE is None:
                return super().depart_image(image)
            try:
                super().dispatch_departure(self.BACKUP_RAW)
            except NotImplementedError:
                self.logger.info("There is no depart visitor for raw")
            image.replace_self([self.BACKUP_IMAGE])
            self.BACKUP_IMAGE, self.BACKUP_RAW = None, None

    SelectHTMLImageTranslator.logger.setLevel(
        sphinx_logger.getEffectiveLevel())
    app.set_translator('html', SelectHTMLImageTranslator)


def setup(app: Sphinx):
    app.connect('builder-inited', extend_default_translator)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': False,
    }
