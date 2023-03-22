# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from typing import Type

from kenning.core.model import ModelWrapper
from kenning.utils.class_loader import get_all_subclasses
from kenning.tests.core.conftest import get_tmp_path
from kenning.tests.core.conftest import remove_file_or_dir
from kenning.tests.core.conftest import get_dataset_random_mock
from kenning.tests.core.conftest import copy_model_to_tmp


MODELWRAPPER_SUBCLASSES = get_all_subclasses(
    'kenning.modelwrappers',
    ModelWrapper,
    raise_exception=True
)


@pytest.fixture(autouse=True, scope='module')
def prepare_models_io_specs():
    for model_cls in MODELWRAPPER_SUBCLASSES:
        if (model_cls.default_dataset is None or
                model_cls.pretrained_modelpath is None):
            continue
        dataset_cls = model_cls.default_dataset
        dataset = get_dataset_random_mock(dataset_cls)

        model_path = model_cls.pretrained_modelpath

        model = model_cls(model_path, dataset, from_file=True)
        model.save_io_specification(model_path)


def create_model(model_cls, dataset):
    if model_cls.pretrained_modelpath is not None:
        model_path = copy_model_to_tmp(model_cls.pretrained_modelpath)
        from_file = True
    else:
        model_path = get_tmp_path()
        remove_file_or_dir(model_path)
        from_file = False
    return model_cls(model_path, dataset, from_file)


@pytest.fixture(scope='function')
def model(request):
    model_cls = request.param

    dataset_cls = model_cls.default_dataset
    dataset = get_dataset_random_mock(dataset_cls)

    return create_model(model_cls, dataset)


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
        _ = create_model(model_cls, None)

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
        dataset_cls = model_cls.default_dataset
        dataset = get_dataset_random_mock(dataset_cls)
        _ = create_model(model_cls, dataset)

    @pytest.mark.parametrize('model', [
        pytest.param(cls, marks=[
            pytest.mark.dependency(
                name=f'test_prepare[{cls.__name__}]',
                depends=[f'test_initializer_with_dataset[{cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestModelWrapper_{cls.__name__}')
        ])
        for cls in MODELWRAPPER_SUBCLASSES
    ], indirect=True)
    def test_prepare(self, model: Type[ModelWrapper]):
        """
        Tests the `prepare_model` method.
        """
        model.prepare_model()

    @pytest.mark.parametrize('model', [
        pytest.param(cls, marks=[
            pytest.mark.dependency(
                depends=[f'test_prepare[{cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestModelWrapper_{cls.__name__}')
        ])
        for cls in MODELWRAPPER_SUBCLASSES
    ], indirect=True)
    def test_save(self, model: Type[ModelWrapper]):
        """
        Tests the `save_model` method.
        """
        model.prepare_model()
        model_save_path = get_tmp_path()
        try:
            model.save_model(model_save_path)
        except NotImplementedError:
            pytest.xfail('save_model not implemented for this model')
        assert model_save_path.exists()

    @pytest.mark.parametrize('model', [
        pytest.param(cls, marks=[
            pytest.mark.dependency(
                depends=[f'test_prepare[{cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestModelWrapper_{cls.__name__}')
        ])
        for cls in MODELWRAPPER_SUBCLASSES
    ], indirect=True)
    def test_inference(self, model: Type[ModelWrapper]):
        """
        Tests the `test_inference` method.
        """
        model.prepare_model()
        try:
            model.test_inference()
        except NotImplementedError:
            pytest.xfail('test_inference not implemented for this model')

    @pytest.mark.parametrize('model', [
        pytest.param(cls, marks=[
            pytest.mark.dependency(
                depends=[f'test_prepare[{cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestModelWrapper_{cls.__name__}')
        ])
        for cls in MODELWRAPPER_SUBCLASSES
    ], indirect=True)
    def test_train(self, model: Type[ModelWrapper]):
        """
        Tests the `train_model` method.
        """
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

    @pytest.mark.parametrize('model', [
        pytest.param(cls, marks=[
            pytest.mark.dependency(
                depends=[f'test_prepare[{cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestModelWrapper_{cls.__name__}')
        ])
        for cls in MODELWRAPPER_SUBCLASSES
    ], indirect=True)
    def test_get_io_spec(self, model: Type[ModelWrapper]):
        """
        Tests the `train_model` method.
        """
        assert model.get_io_specification_from_model() is not None

    @pytest.mark.parametrize('model', [
        pytest.param(cls, marks=[
            pytest.mark.dependency(
                depends=[f'test_prepare[{cls.__name__}]']
            ),
            pytest.mark.xdist_group(name=f'TestModelWrapper_{cls.__name__}')
        ])
        for cls in MODELWRAPPER_SUBCLASSES
    ], indirect=True)
    def test_get_framework_and_version(self, model: Type[ModelWrapper]):
        """
        Tests the `train_model` method.
        """
        assert model.get_framework_and_version() is not None
