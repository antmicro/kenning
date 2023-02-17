"""
Module with sphinx extension, which swap format of image in figure node
if path extensions equals .*
"""
import copy
import logging
from pathlib import Path
from docutils import nodes
from sphinx.util.docutils import SphinxTranslator
from sphinx.application import Sphinx, logger as sphinx_logger

_LOGGER = logging.getLogger(__name__)
_BACKUP_IMAGE, _BACKUP_RAW = None, None


def visit_figure_image_html(self: SphinxTranslator, figure: nodes.figure):
    """
    Function executed when node is visited - right before builder
    (for HTML files) write this node to file in proper format. It swaps
    image with raw html node if there is avaiable image in HTML format.

    Parameters
    ----------
    self : SphinxTranslator
        This function is executed as one of methods from self - instance of
        SphinxTranslator
    figure : nodes.figure
        Processed node

    Note
    ----
    To preserve original state of nodes (e.g. for other builds), this function
    save state of image node to global variable _BACKUP_IMAGE and the orginal
    state of node is restored during departure - function
    departure_swap_image_html
    """
    # Searching for image node
    image: nodes.image = None
    for child in figure.children:
        if isinstance(child, nodes.image):
            image = child
            break
    if image is None:
        _LOGGER.debug("Figure node don't have an image")
        return
    if 'candidates' not in image or len(image['candidates']) == 1:
        _LOGGER.info("Figure node have at most one candidate - "
                     "it's not possible to check other avaiable formats")
        return
    if 'image/x-html' not in image['candidates']:
        _LOGGER.debug("There is no HTML candidate for this image")
        return
    # Backup image
    global _BACKUP_IMAGE, _BACKUP_RAW
    _BACKUP_IMAGE = copy.deepcopy(image)
    # Create raw node from image data
    filepath = image['candidates']['image/x-html']
    with open(Path(self.settings.env.srcdir, filepath), 'r') as fd:
        html = fd.read()
    raw = nodes.raw('', text=html, format='html')
    _BACKUP_RAW = raw
    # Replacing image with raw
    figure.replace(image, [raw])


def departure_figure_image_html(self, figure: nodes.figure):
    """
    Function executed when visit in node ended - right after builder
    (for HTML files) wrote this node to file in proper format. It swaps raw
    html with backuped image node.

    Parameters
    ----------
    self : SphinxTranslator
        This function is executed as one of methods from self - instance of
        SphinxTranslator
    figure : nodes.figure
        Processed node
    """
    # Append missing ending of figcaption
    self.body.append('</figcaption>')
    # Checking if there is node to restore
    global _BACKUP_IMAGE, _BACKUP_RAW
    if _BACKUP_IMAGE is None or _BACKUP_RAW is None or \
            all([child != _BACKUP_RAW for child in figure.children]):
        _LOGGER.debug("There is no image to restore")
        return
    # Replacing raw node with image
    figure.replace(_BACKUP_RAW, _BACKUP_IMAGE)
    _BACKUP_IMAGE, _BACKUP_RAW = None, None


def setup(app: Sphinx):
    _LOGGER.setLevel(sphinx_logger.getEffectiveLevel())
    app.add_node(nodes.figure, override=False, html=(
        visit_figure_image_html, departure_figure_image_html))

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': False,
    }
