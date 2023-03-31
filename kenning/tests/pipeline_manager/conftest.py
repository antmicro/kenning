import pytest
from pytest_mock import MockerFixture

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.optimizer import Optimizer
from kenning.core.runtime import Runtime

from kenning.utils.class_loader import get_all_subclasses


def create_mocks(mocker, module_name, core_cls):
    subclasses = get_all_subclasses(
        module_name,
        core_cls
    )
    for cls in subclasses:
        cls_name = f"{cls.__module__}.{cls.__name__}"
        try:
            static_io_spec_getter = cls.parse_io_specification_from_json
        except AttributeError:
            static_io_spec_getter = None
        mock = mocker.patch(cls_name)
        mock.__name__ = cls.__name__
        mocker.patch(f"{cls_name}.from_json", return_value=mock)
        if static_io_spec_getter is not None:
            mock.parse_io_specification_from_json = static_io_spec_getter


@pytest.fixture(scope='function')
def mock_enviroment(mocker: MockerFixture):
    """
    Creates mocks of every implementation of Kenning's runtime, compiler,
    modelwrapper and dataset module
    """
    create_mocks(mocker, 'kenning.runtimes', Runtime)
    create_mocks(mocker, 'kenning.compilers', Optimizer)
    create_mocks(mocker, 'kenning.modelwrappers', ModelWrapper)
    create_mocks(mocker, "kenning.datasets", Dataset)
