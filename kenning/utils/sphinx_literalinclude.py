# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Sphinx extension overriding `literalinclude` directive.
"""

from sphinx.application import Sphinx
from sphinx.directives.code import LiteralInclude as _LiteralInclude


class LiteralInclude(_LiteralInclude):
    """
    `literalinclude` directive ignoring `save-as` argument
    and using it for caption.
    """

    def run(self):
        # Remove save-as argument
        save_as = None
        if "save-as" in self.arguments[0]:
            save_as = self.arguments[0][self.arguments[0].find("save-as") :]
            self.arguments[0] = self.arguments[0][: -len(save_as) - 1]
        # If caption is not set, use file name from save-as
        if save_as and not (
            "caption" in self.options and self.options["caption"]
        ):
            self.options["caption"] = save_as.split("=", 1)[1].strip()
        return super().run()


def setup(app: Sphinx):
    """
    Installs literalinclude directive in Sphinx.
    """
    app.add_directive("literalinclude", LiteralInclude, override=True)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
    }
