import pytest
from typing import Type, Final, Tuple

from kenning.core.optimizer import Optimizer
from kenning.core.optimizer import ConversionError
from kenning.core.optimizer import CompilationError
from kenning.core.model import ModelWrapper
from kenning.utils.class_loader import get_all_subclasses
from kenning.tests.core.conftest import get_tmp_path
from kenning.tests.core.conftest import get_default_dataset_model
from kenning.tests.core.conftest import UnknownFramework


OPTIMIZER_SUBCLASSES: Final = get_all_subclasses(
    'kenning.compilers',
    Optimizer,
    raise_exception=True
)

OPTIMIZER_INPUTTYPES: Final = [
    (opt, inp) for opt in OPTIMIZER_SUBCLASSES for inp in opt.inputtypes
]


def prepare_objects(
        opt_cls: Type[Optimizer],
        inputtype: str) -> Tuple[Optimizer, ModelWrapper]:
    compiled_model_path = get_tmp_path()
    try:
        dataset, model = get_default_dataset_model(inputtype)
    except UnknownFramework:
        pytest.xfail(f'Unknown framework: {inputtype}')

    optimizer = opt_cls(dataset, compiled_model_path)
    optimizer.set_input_type(inputtype)

    return optimizer, model


class TestOptimizer:
    @pytest.mark.parametrize('opt_cls,inputtype', [
        pytest.param(opt_cls, inputtype, marks=[
            pytest.mark.dependency(
                name=f'test_initializer[{opt_cls.__name__},{inputtype}]'
            ),
            pytest.mark.xdist_group(name=f'TestOptimizer_{opt_cls.__name__}')
        ])
        for opt_cls, inputtype in OPTIMIZER_INPUTTYPES
    ])
    def test_initializer(self, opt_cls: Type[Optimizer], inputtype: str):
        """
        Tests optimizer initialization.
        """
        _ = prepare_objects(opt_cls, inputtype)

    @pytest.mark.parametrize('opt_cls,inputtype', [
        pytest.param(opt_cls, inputtype, marks=[
            pytest.mark.dependency(
                depends=[f'test_initializer[{opt_cls.__name__},{inputtype}]']
            ),
            pytest.mark.xdist_group(name=f'TestOptimizer_{opt_cls.__name__}')
        ])
        for opt_cls, inputtype in OPTIMIZER_INPUTTYPES
    ])
    def test_compilation(self, opt_cls: Type[Optimizer], inputtype: str):
        """
        Tests optimizer compilation.
        """
        optimizer, model = prepare_objects(opt_cls, inputtype)
        try:
            optimizer.compile(model.modelpath)
            assert optimizer.compiled_model_path.exists()
        except CompilationError as e:
            pytest.xfail(f'compilation error {e}')
        except ConversionError as e:
            pytest.xfail(f'conversion error {e}')

    @pytest.mark.parametrize('opt_cls,inputtype', [
        pytest.param(opt_cls, inputtype, marks=[
            pytest.mark.dependency(
                depends=[f'test_initializer[{opt_cls.__name__},{inputtype}]']
            ),
            pytest.mark.xdist_group(name=f'TestOptimizer_{opt_cls.__name__}')
        ])
        for opt_cls, inputtype in OPTIMIZER_INPUTTYPES
    ])
    def test_get_framework_and_version(
            self,
            opt_cls: Type[Optimizer],
            inputtype: str):
        """
        Tests `get_framework_and_version` method.
        """
        optimizer, _ = prepare_objects(opt_cls, inputtype)

        assert optimizer.get_framework_and_version() is not None
