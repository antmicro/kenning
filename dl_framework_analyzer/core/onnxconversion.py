from collections import namedtuple
from pathlib import Path
from enum import Enum
import onnx


ModelEntry = namedtuple('ModelEntry', ['name', 'modelobj', 'expectedinput'])
Support = namedtuple('Support', ['framework', 'model', 'export', 'import'])


class SupportStatus(Enum):
    NOTIMPLEMENTED = 0
    SUPPORTED = 1
    UNSUPPORTED = 2
    UNVERIFIED = 3
    ERROR = 4
    ONNXMODELINVALID = 5


class ONNXConversion(object):
    """
    Creates ONNX conversion support matrix for given framework and models.
    """

    def __init__(self, framework, version, modelslist):
        self.modelslist = modelslist
        self.framework = framework
        self.version = version

    def add_entry(self, name, modelobj, expectedinput):
        self.modelslist.append(ModelEntry(name, modelobj, expectedinput))

    def onnx_export(self, modelentry: ModelEntry, exportpath: Path):
        raise NotImplementedError

    def onnx_import(self, modelentry: ModelEntry, importpath: Path):
        raise NotImplementedError

    def _onnx_export(self, modelentry: ModelEntry, exportpath: Path):
        try:
            self.onnx_export(modelentry, exportpath)
        except NotImplementedError:
            return SupportStatus.NOTIMPLEMENTED
        except:
            return SupportStatus.ERROR

    def _onnx_import(self, modelentry: ModelEntry, importpath: Path):
        try:
            self.onnx_import(modelentry, exportpath)
        except NotImplementedError:
            return SupportStatus.NOTIMPLEMENTED
        except:
            return SupportStatus.ERROR

    def check_conversions(self, modelsdir: Path):
        modelsdir = Path(modelsdir)
        modelsdir.mkdir(parents=True, exist_ok=True)
        supportlist = []
        for modelentry in self.modelslist:
            onnxtargetpath = modelsdir / f'{modelentry.name}.onnx'
            exported = self.onnx_export(modelentry, onnxtargetpath)
            if exported:
                onnxmodel = onnx.load(onnxtargetpath)
                try:
                    onnx.checker.check_model(onnxmodel)
                except:
                    exported = SupportStatus.ONNXMODELINVALID
            imported = SupportStatus.NOTIMPLEMENTED
            if exported != SupportStatus.SUPPORTED:
                imported = self._onnx_import(modelentry, onnxtargetpath)
            supportlist.append(Support(
                self.framework,
                modelentry.name,
                exported,
                imported
            ))
        return supportlist
