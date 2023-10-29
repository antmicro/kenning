# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Provides an API for ONNX conversions of various models in a given framework.
"""

from collections import namedtuple
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import List, Dict, Callable, Any

import onnx

from kenning.utils.logger import KLogger

ModelEntry = namedtuple("ModelEntry", ["name", "modelgenerator", "parameters"])
ModelEntry.__doc__ = """
Represents single model entry for a given framework.

Attributes
----------
name : str
    Name of the model.
modelgenerator : FuncVar
    Function variable that can be called without any parameters to create
    the model.
parameters : Dict[str, Any]
    The dictionary with additional model conversion/initialization parameters.
"""

Support = namedtuple(
    "Support", ["framework", "version", "model", "exported", "imported"]
)
Support.__doc__ = """
Shows the framework's ONNX conversion support status for a given model.

Attributes
----------
framework : str
    Name of the framework.
version : str
    Version of the framework.
model : str
    Name of the model.
exported : SupportStatus
    The status of exporting the model to the ONNX.
imported : SupportStatus
    The status of importing the model from the ONNX.
"""


class SupportStatus(Enum):
    """
    Enum representing the support status for ONNX conversion.

    NOTIMPLEMENTED - import/export is not implemented.
    SUPPORTED - import/export is supported.
    UNSUPPORTED - import/export is not supported.
    UNVERIFIED - import/export is unverified (due to lack of model to process).
    ERROR - import/export resulted in an error.
    ONNXMODELINVALID - exported ONNX model is invalid.
    """

    NOTIMPLEMENTED = 0
    SUPPORTED = 1
    UNSUPPORTED = 2
    UNVERIFIED = 3
    ERROR = 4
    ONNXMODELINVALID = 5

    def __str__(self):
        converter = {
            self.NOTIMPLEMENTED: "not implemented",
            self.SUPPORTED: "supported",
            self.UNSUPPORTED: "unsupported",
            self.UNVERIFIED: "unverified",
            self.ERROR: "ERROR",
            self.ONNXMODELINVALID: "Converter returned invalid ONNX model",
        }
        return converter[SupportStatus(self.value)]


class ONNXConversion(ABC):
    """
    Creates ONNX conversion support matrix for given framework and models.
    """

    def __init__(self, framework: str, version: str):
        """
        Prepares structures for ONNX conversion.

        The framework and version values should be provided by the inheriting
        classes.

        Parameters
        ----------
        framework : str
            Name of the framework.
        version : str
            Version of the framework (should be derived from __version__).
        """
        self.modelslist = []
        self.framework = framework
        self.version = version
        self.prepare()

    def add_entry(
        self, name: str, modelgenerator: Callable, **kwargs: Dict[str, Any]
    ):
        """
        Adds new model for verification.

        Parameters
        ----------
        name : str
            Full name of the model, should match the name of the same models
            in other framework's implementations.
        modelgenerator : Callable
            Function that generates the model for ONNX conversion in a given
            framework. The callable should accept no arguments.
        **kwargs : Dict[str, Any]
            Additional arguments that are passed to ModelEntry object as
            parameters.
        """
        self.modelslist.append(ModelEntry(name, modelgenerator, kwargs))

    @abstractmethod
    def onnx_export(
        self, modelentry: ModelEntry, exportpath: Path
    ) -> SupportStatus:
        """
        Virtual function for exporting the model to ONNX in a given framework.

        This method needs to be implemented for a given framework in inheriting
        class.

        Parameters
        ----------
        modelentry : ModelEntry
            ModelEntry object.
        exportpath : Path
            Path to the output ONNX file.

        Returns
        -------
        SupportStatus
            The support status of exporting given model to ONNX.
        """
        ...

    @abstractmethod
    def onnx_import(
        self, modelentry: ModelEntry, importpath: Path
    ) -> SupportStatus:
        """
        Virtual function for importing ONNX model to a given framework.

        This method needs to be implemented for a given framework in inheriting
        class.

        Parameters
        ----------
        modelentry : ModelEntry
            ModelEntry object.
        importpath : Path
            Path to the input ONNX file.

        Returns
        -------
        SupportStatus
            The support status of importing given model from ONNX.
        """
        ...

    @abstractmethod
    def prepare(self):
        """
        Virtual function for preparing the ONNX conversion test.

        This method should add model entries using add_entry methods.

        It is later called in the constructor to prepare the list of models to
        test.
        """
        ...

    def _onnx_export(self, modelentry: ModelEntry, exportpath: Path):
        try:
            return self.onnx_export(modelentry, exportpath)
        except NotImplementedError:
            return SupportStatus.NOTIMPLEMENTED
        except Exception as e:
            KLogger.error(e, stack_info=True)
            return SupportStatus.ERROR

    def _onnx_import(self, modelentry: ModelEntry, importpath: Path):
        try:
            return self.onnx_import(modelentry, importpath)
        except NotImplementedError:
            return SupportStatus.NOTIMPLEMENTED
        except Exception as e:
            KLogger.error(e, stack_info=True)
            return SupportStatus.ERROR

    def check_conversions(self, modelsdir: Path) -> List[Support]:
        """
        Runs ONNX conversion for every model entry in the list of models.

        Parameters
        ----------
        modelsdir : Path
            Path to the directory where the intermediate models will be saved.

        Returns
        -------
        List[Support]
            List with Support tuples describing support status.
        """
        KLogger.info(f"~~~~> {self.framework} (ver. {self.version})")
        modelsdir = Path(modelsdir)
        modelsdir.mkdir(parents=True, exist_ok=True)
        supportlist = []
        for modelentry in self.modelslist:
            onnxtargetpath = modelsdir / f"{modelentry.name}.onnx"
            KLogger.info(f"\t{modelentry.name} ===> {onnxtargetpath}")
            KLogger.info("\t\tExporting...")
            exported = self._onnx_export(modelentry, onnxtargetpath)
            KLogger.info("\t\tExported")
            if exported == SupportStatus.SUPPORTED:
                KLogger.info("\t\t\tVerifying...")
                onnxmodel = onnx.load(onnxtargetpath)
                try:
                    onnx.checker.check_model(onnxmodel)
                    KLogger.info("\t\t\tVerified")
                except Exception as ex:
                    KLogger.error(ex, stack_info=True)
                    exported = SupportStatus.ONNXMODELINVALID
            imported = SupportStatus.UNVERIFIED
            if exported == SupportStatus.SUPPORTED:
                KLogger.info("\t\tImporting...")
                imported = self._onnx_import(modelentry, onnxtargetpath)
                KLogger.info("\t\tImported")
            supportlist.append(
                Support(
                    self.framework,
                    self.version,
                    modelentry.name,
                    exported,
                    imported,
                )
            )
        return supportlist
