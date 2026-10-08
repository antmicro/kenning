# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Reads a project's requirements and extras from its pyproject.toml.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

RE_EXACT_PYTHON = re.compile(r";.*python_version\s*==")


class Pyproject:
    """
    Requirements and extras of a project described by pyproject.toml.
    """

    def __init__(self, pyproject: Dict):
        """
        Reads requirements from parsed pyproject.toml.

        Parameters
        ----------
        pyproject : Dict
            Parsed pyproject.toml
        """
        project = pyproject["project"]
        self.name: str = project["name"]
        self.dependencies: List[str] = project.get("dependencies", [])
        self.optional_dependencies: Dict[str, List[str]] = project.get(
            "optional-dependencies", {}
        )

    @classmethod
    def load(cls, path: Path) -> "Pyproject":
        """
        Loads pyproject.toml.

        Parameters
        ----------
        path : Path
            Path to pyproject.toml

        Returns
        -------
        Pyproject
            Requirements of the project
        """
        with open(path, "rb") as fd:
            return cls(tomllib.load(fd))

    def get_extras(self) -> Dict[str, List[str]]:
        """
        Lists requirements of the base installation and of each extra.

        Returns
        -------
        Dict[str, List[str]]
            Mapping from extra name to requirements, with base requirements
            under the project name
        """
        extras = {self.name: list(self.dependencies)}
        for extra in self.optional_dependencies:
            extras[extra] = list(
                dict.fromkeys(self.dependencies + self._expand(extra, set()))
            )
        return extras

    def _own_extras(self, requirement: str) -> List[str]:
        """
        Lists extras of this project referenced by a requirement.

        Parameters
        ----------
        requirement : str
            Requirement, e.g. `kenning[torch]`

        Returns
        -------
        List[str]
            Referenced extras, empty if the requirement is not
            a reference to this project
        """
        parsed = Requirement(requirement)
        if (
            canonicalize_name(parsed.name) != canonicalize_name(self.name)
            or parsed.url
        ):
            return []
        return sorted(parsed.extras)

    @staticmethod
    def _add_marker(requirement: str, marker: Optional[Marker]) -> str:
        """
        Restricts a requirement with an additional environment marker.

        Parameters
        ----------
        requirement : str
            Requirement, e.g. `torch>=2.0 ; python_version < '3.13'`
        marker : Optional[Marker]
            Marker to combine with the requirement's own one, if any

        Returns
        -------
        str
            Requirement that applies only where both markers hold
        """
        if marker is None:
            return requirement
        parsed = Requirement(requirement)
        if parsed.marker is not None:
            marker = Marker(f"({parsed.marker}) and ({marker})")
        parsed.marker = marker
        return str(parsed)

    def _expand(self, extra: str, seen: Set[str]) -> List[str]:
        """
        Lists requirements of an extra, with references to other extras
        of this project replaced by their requirements.

        Parameters
        ----------
        extra : str
            Extra name
        seen : Set[str]
            Extras already expanded, to break reference cycles

        Returns
        -------
        List[str]
            Requirements of the extra
        """
        if extra in seen:
            return []
        seen.add(extra)
        requirements = []
        for requirement in self.optional_dependencies[extra]:
            nested = self._own_extras(requirement)
            if not nested:
                requirements.append(requirement)
                continue
            marker = Requirement(requirement).marker
            for nested_extra in nested:
                requirements += [
                    self._add_marker(line, marker)
                    for line in self._expand(nested_extra, seen)
                ]
        return requirements
