import pytest
from typing import Type, Final

from kenning.core.runtime import Runtime
from kenning.runtimes import *  # noqa: 401, 403
from kenning.tests.core.conftest import get_all_subclasses
from kenning.tests.core.conftest import get_default_dataset_model
from kenning.tests.core.conftest import UnknownFramework


RUNTIME_SUBCLASSES: Final = get_all_subclasses(Runtime)

RUNTIME_INPUTTYPES: Final = [
    (run, inp) for run in RUNTIME_SUBCLASSES for inp in run.inputtypes
]


class TestRuntime:

    @pytest.mark.parametrize('runtime_cls,inputtype', [
        pytest.param(runtime_cls, inputtype, marks=[
            pytest.mark.dependency(
                name=f'test_initializer[{runtime_cls.__name__}]'
            ),
            pytest.mark.xdist_group(name=f'TestRuntime_{runtime_cls.__name__}')
        ])
        for runtime_cls, inputtype in RUNTIME_INPUTTYPES
    ])
    def test_initializer(self, runtime_cls: Type[Runtime], inputtype: str):

        try:
            _, model = get_default_dataset_model(inputtype)
            _ = runtime_cls(protocol=None, modelpath=model)
        except UnknownFramework:
            pytest.xfail(f'Unknown framework: {inputtype}')
        except Exception as e:
            pytest.fail(f'Exception {e}')

    @pytest.mark.parametrize('runtime_cls,inputtype', [
        pytest.param(runtime_cls, inputtype, marks=[
            pytest.mark.dependency(
                depends=[f'test_initializer[{runtime_cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestRuntime_{runtime_cls.__name__}')
        ])
        for runtime_cls, inputtype in RUNTIME_INPUTTYPES
    ])
    def test_inference(self, runtime_cls: Type[Runtime], inputtype: str):
        try:
            dataset, model = get_default_dataset_model(inputtype)
            runtime = runtime_cls(None, modelpath=model.modelpath)

            runtime.run_locally(dataset, model, str(model.modelpath))

        except UnknownFramework:
            pytest.xfail(f'Unknown framework: {inputtype}')
        except Exception as e:
            pytest.fail(f'Exception {e}')
