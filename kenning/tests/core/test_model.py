import pytest
from typing import Type, Final
import os

from kenning.core.model import ModelWrapper
from kenning.modelwrappers.classification import *  # noqa: 401, 403
from kenning.modelwrappers.detectors import *  # noqa: 401, 403
from kenning.modelwrappers.instance_segmentation import *  # noqa: 401, 403
from kenning.tests.core.conftest import get_tmp_path
from kenning.tests.core.conftest import get_all_subclasses
from kenning.tests.core.conftest import remove_file_or_dir
from kenning.tests.core.conftest import get_dataset_random_mock


MODELWRAPPER_SUBCLASSES: Final = get_all_subclasses(ModelWrapper)


class TestModelWrapper:

    @pytest.mark.parametrize('model_cls', [
        pytest.param(cls, marks=[
            pytest.mark.xdist_group(name=f'TestModelWrapper_{cls.__name__}')
        ])
        for cls in MODELWRAPPER_SUBCLASSES
    ])
    def test_initializer_no_dataset(self, model_cls: Type[ModelWrapper]):
        """
        Tests model initialization without specified dataset.
        """
        modelpath = get_tmp_path()
        remove_file_or_dir(modelpath)

        if model_cls.pretrained_modelpath is not None:
            model_path = model_cls.pretrained_modelpath
            from_file = True
        else:
            model_path = modelpath
            from_file = False

        _ = model_cls(model_path, None, from_file)

    @pytest.mark.parametrize('model_cls', [
        pytest.param(cls, marks=[
            pytest.mark.dependency(
                name=f'test_initializer_with_dataset[{cls.__name__}]'
            ),
            pytest.mark.xdist_group(name=f'TestModelWrapper_{cls.__name__}')
        ])
        for cls in MODELWRAPPER_SUBCLASSES
    ])
    def test_initializer_with_dataset(self, model_cls: Type[ModelWrapper]):
        """
        Tests model initialization with specified dataset.
        """
        modelpath = get_tmp_path()
        remove_file_or_dir(modelpath)

        dataset_cls = model_cls.default_dataset
        dataset = get_dataset_random_mock(dataset_cls)

        if model_cls.pretrained_modelpath is not None:
            model_path = model_cls.pretrained_modelpath
            from_file = True
        else:
            model_path = modelpath
            from_file = False

        _ = model_cls(model_path, dataset, from_file)

    @pytest.mark.parametrize('model_cls', [
        pytest.param(cls, marks=[
            pytest.mark.dependency(
                name=f'test_prepare[{cls.__name__}]',
                depends=[f'test_initializer_with_dataset[{cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestModelWrapper_{cls.__name__}')
        ])
        for cls in MODELWRAPPER_SUBCLASSES
    ])
    def test_prepare(self, model_cls: Type[ModelWrapper]):
        """
        Tests the `prepare_model` method.
        """
        modelpath = get_tmp_path()
        remove_file_or_dir(modelpath)

        dataset_cls = model_cls.default_dataset
        dataset = get_dataset_random_mock(dataset_cls)

        if model_cls.pretrained_modelpath is not None:
            model_path = model_cls.pretrained_modelpath
            from_file = True
        else:
            model_path = modelpath
            from_file = False

        model = model_cls(model_path, dataset, from_file)
        model.prepare_model()

    @pytest.mark.parametrize('model_cls', [
        pytest.param(cls, marks=[
            pytest.mark.dependency(
                depends=[f'test_prepare[{cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestModelWrapper_{cls.__name__}')
        ])
        for cls in MODELWRAPPER_SUBCLASSES
    ])
    def test_save(self, model_cls: Type[ModelWrapper]):
        """
        Tests the `save_model` method.
        """
        modelpath = get_tmp_path()
        remove_file_or_dir(modelpath)

        dataset_cls = model_cls.default_dataset
        dataset = get_dataset_random_mock(dataset_cls)

        if model_cls.pretrained_modelpath is not None:
            model_path = model_cls.pretrained_modelpath
            from_file = True
        else:
            model_path = modelpath
            from_file = False

        model = model_cls(model_path, dataset, from_file)
        model.prepare_model()
        model_save_path = get_tmp_path()
        try:
            model.save_model(model_save_path)
        except NotImplementedError:
            pytest.xfail('save_model not implemented for this model')
        assert (os.path.isfile(model_save_path) or
                os.path.isdir(model_save_path))

    @pytest.mark.parametrize('model_cls', [
        pytest.param(cls, marks=[
            pytest.mark.dependency(
                depends=[f'test_prepare[{cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestModelWrapper_{cls.__name__}')
        ])
        for cls in MODELWRAPPER_SUBCLASSES
    ])
    def test_inference(self, model_cls: Type[ModelWrapper]):
        """
        Tests the `test_inference` method.
        """
        modelpath = get_tmp_path()
        remove_file_or_dir(modelpath)

        dataset_cls = model_cls.default_dataset
        dataset = get_dataset_random_mock(dataset_cls)

        if model_cls.pretrained_modelpath is not None:
            model_path = model_cls.pretrained_modelpath
            from_file = True
        else:
            model_path = modelpath
            from_file = False

        model = model_cls(model_path, dataset, from_file)
        model.prepare_model()
        try:
            model.test_inference()
        except NotImplementedError:
            pytest.xfail('test_inference not implemented for this model')

    @pytest.mark.parametrize('model_cls', [
        pytest.param(cls, marks=[
            pytest.mark.dependency(
                depends=[f'test_prepare[{cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestModelWrapper_{cls.__name__}')
        ])
        for cls in MODELWRAPPER_SUBCLASSES
    ])
    def test_train(self, model_cls: Type[ModelWrapper]):
        """
        Tests the `train_model` method.
        """
        modelpath = get_tmp_path()
        remove_file_or_dir(modelpath)

        dataset_cls = model_cls.default_dataset
        dataset = get_dataset_random_mock(dataset_cls)

        if model_cls.pretrained_modelpath is not None:
            model_path = model_cls.pretrained_modelpath
            from_file = True
        else:
            model_path = modelpath
            from_file = False

        model = model_cls(model_path, dataset, from_file)
        model.prepare_model()
        try:
            model.train_model(
                batch_size=16,
                learning_rate=.01,
                epochs=1,
                logdir=r'/tmp/logdir'
            )
        except NotImplementedError:
            pytest.xfail('train_model not implemented for this model')

    @pytest.mark.parametrize('model_cls', [
        pytest.param(cls, marks=[
            pytest.mark.dependency(
                depends=[f'test_prepare[{cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestModelWrapper_{cls.__name__}')
        ])
        for cls in MODELWRAPPER_SUBCLASSES
    ])
    def test_get_io_spec(self, model_cls: Type[ModelWrapper]):
        """
        Tests the `train_model` method.
        """
        modelpath = get_tmp_path()
        remove_file_or_dir(modelpath)

        dataset_cls = model_cls.default_dataset
        dataset = get_dataset_random_mock(dataset_cls)

        if model_cls.pretrained_modelpath is not None:
            model_path = model_cls.pretrained_modelpath
            from_file = True
        else:
            model_path = modelpath
            from_file = False

        model = model_cls(model_path, dataset, from_file)
        assert model.get_io_specification_from_model() is not None

    @pytest.mark.parametrize('model_cls', [
        pytest.param(cls, marks=[
            pytest.mark.dependency(
                depends=[f'test_prepare[{cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestModelWrapper_{cls.__name__}')
        ])
        for cls in MODELWRAPPER_SUBCLASSES
    ])
    def test_get_framework_and_version(self, model_cls: Type[ModelWrapper]):
        """
        Tests the `train_model` method.
        """
        modelpath = get_tmp_path()
        remove_file_or_dir(modelpath)

        dataset_cls = model_cls.default_dataset
        dataset = get_dataset_random_mock(dataset_cls)

        if model_cls.pretrained_modelpath is not None:
            model_path = model_cls.pretrained_modelpath
            from_file = True
        else:
            model_path = modelpath
            from_file = False

        model = model_cls(model_path, dataset, from_file)
        assert model.get_framework_and_version() is not None
