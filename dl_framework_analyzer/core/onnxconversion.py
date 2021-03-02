from collections import namedtuple
from pathlib import Path
from enum import Enum
import onnx

from dl_framework_analyzer.utils.logger import get_logger

ModelEntry = namedtuple('ModelEntry', ['name', 'modelgenerator', 'parameters'])
Support = namedtuple('Support', ['framework', 'version', 'model', 'exported', 'imported'])


class SupportStatus(Enum):
    NOTIMPLEMENTED = 0
    SUPPORTED = 1
    UNSUPPORTED = 2
    UNVERIFIED = 3
    ERROR = 4
    ONNXMODELINVALID = 5
    NOTPROVIDED = 6

    def __str__(self):
        converter = {
            self.NOTIMPLEMENTED: 'not implemented',
            self.SUPPORTED: 'supported',
            self.UNSUPPORTED: 'unsupported',
            self.UNVERIFIED: 'unverified',
            self.ERROR: 'ERROR',
            self.ONNXMODELINVALID: 'Converter returned invalid ONNX model',
            self.NOTPROVIDED: 'Not provided'
        }
        return converter[SupportStatus(self.value)]


class ONNXConversion(object):
    """
    Creates ONNX conversion support matrix for given framework and models.
    """

    def __init__(self, framework, version):
        self.modelslist = []
        self.framework = framework
        self.version = version
        self.logger = get_logger()
        self.prepare()

    def add_entry(self, name, modelgenerator, **kwargs):
        self.modelslist.append(ModelEntry(name, modelgenerator, kwargs))

    def onnx_export(self, modelentry: ModelEntry, exportpath: Path):
        raise NotImplementedError

    def onnx_import(self, modelentry: ModelEntry, importpath: Path):
        raise NotImplementedError

    def prepare(self):
        raise NotImplementedError

    def _onnx_export(self, modelentry: ModelEntry, exportpath: Path):
        try:
            self.logger.info(f'    {modelentry.name} ===> {exportpath}')
            return self.onnx_export(modelentry, exportpath)
        except NotImplementedError:
            return SupportStatus.NOTIMPLEMENTED
        except Exception as e:
            print(e)
            return SupportStatus.ERROR

    def _onnx_import(self, modelentry: ModelEntry, importpath: Path):
        try:
            self.onnx_import(modelentry, importpath)
        except NotImplementedError:
            return SupportStatus.NOTIMPLEMENTED
        except Exception as e:
            print(e)
            return SupportStatus.ERROR

    def check_conversions(self, modelsdir: Path):
        self.logger.info(f'~~~~> {self.framework} (ver. {self.version})')
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
            imported = SupportStatus.UNVERIFIED
            if exported == SupportStatus.SUPPORTED:
                imported = self._onnx_import(modelentry, onnxtargetpath)
            supportlist.append(Support(
                self.framework,
                self.version,
                modelentry.name,
                exported,
                imported
            ))
        return supportlist
