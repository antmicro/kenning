# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Sphinx extension overriding `code-block` directive.
"""

from sphinx.application import Sphinx
from sphinx.directives.code import CodeBlock as _CodeBlock


class CodeBlock(_CodeBlock):
    """
    `code-block` directive ignoring `save-as` argument
    and using it for caption.
    """

    optional_arguments = 2

    def run(self):
        # Remove save-as argument
        save_as = None
        for i, arg in enumerate(self.arguments):
            if not arg.startswith("save-as"):
                continue
            save_as = self.arguments.pop(i)
            break
        # If caption is not set, use file name from save-as
        if save_as and not (
            "caption" in self.options and self.options["caption"]
        ):
            self.options["caption"] = save_as.split("=", 1)[1].strip()
        return super().run()


def setup(app: Sphinx):
    """
    Installs code-block directive in Sphinx.
    """
    app.add_directive("code-block", CodeBlock, override=True)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
    }
