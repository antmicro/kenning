import pytest
from typing import Type, Final, Tuple

from kenning.core.runtime import Runtime
from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.utils.class_loader import get_all_subclasses
from kenning.tests.core.conftest import get_default_dataset_model
from kenning.tests.core.conftest import UnknownFramework


RUNTIME_SUBCLASSES: Final = get_all_subclasses(
    'kenning.runtimes',
    Runtime,
    raise_exception=True
)

RUNTIME_INPUTTYPES: Final = [
    (run, inp) for run in RUNTIME_SUBCLASSES for inp in run.inputtypes
]


def prepare_objects(
        runtime_cls: Type[Runtime],
        inputtype: str) -> Tuple[Runtime, Dataset, ModelWrapper]:
    try:
        dataset, model = get_default_dataset_model(inputtype)
    except UnknownFramework:
        pytest.xfail(f'Unknown framework: {inputtype}')

    runtime = runtime_cls(protocol=None, modelpath=model.modelpath)

    return runtime, dataset, model


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
        """
        Tests runtime initialization.
        """
        _ = prepare_objects(runtime_cls, inputtype)

    @pytest.mark.parametrize('runtime_cls,inputtype', [
        pytest.param(runtime_cls, inputtype, marks=[
            pytest.mark.dependency(
                name=f'test_prepare_local[{runtime_cls.__name__}]',
                depends=[f'test_initializer[{runtime_cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestRuntime_{runtime_cls.__name__}')
        ])
        for runtime_cls, inputtype in RUNTIME_INPUTTYPES
    ])
    def test_prepare_local(self, runtime_cls: Type[Runtime], inputtype: str):
        """
        Tests the `preprocess_input` method.
        """
        runtime, _, _ = prepare_objects(runtime_cls, inputtype)

        assert runtime.prepare_local()

    @pytest.mark.parametrize('runtime_cls,inputtype', [
        pytest.param(runtime_cls, inputtype, marks=[
            pytest.mark.dependency(
                depends=[f'test_prepare_local[{runtime_cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestRuntime_{runtime_cls.__name__}')
        ])
        for runtime_cls, inputtype in RUNTIME_INPUTTYPES
    ])
    def test_inference(self, runtime_cls: Type[Runtime], inputtype: str):
        """
        Tests the `run_locally` method.
        """
        runtime, dataset, model = prepare_objects(runtime_cls, inputtype)

        runtime.run_locally(dataset, model, str(model.modelpath))
