# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Type
from unittest.mock import patch

import pytest

from kenning.core.dataset import Dataset
from kenning.core.exceptions import ModelNotLoadedError, NotSupportedError
from kenning.core.model import ModelWrapper
from kenning.modelwrappers.anomaly_detection.ai8x_cnn import (
    Ai8xAnomalyDetectionCNN,
)
from kenning.modelwrappers.llm.llm import LLM
from kenning.tests.core.conftest import (
    copy_model_to_tmp,
    get_dataset_random_mock,
    get_tmp_path,
    remove_file_or_dir,
)
from kenning.utils.class_loader import get_all_subclasses
from kenning.utils.resource_manager import ResourceManager, ResourceURI

MODELWRAPPER_SUBCLASSES = get_all_subclasses(
    "kenning.modelwrappers", ModelWrapper, raise_exception=True
)

# Remove LLM from the list of modelwrappers to test
# TODO: Those should be tested in a separate test suite.
# TODO: Add tests for minimal LLM model
MODELWRAPPER_SUBCLASSES = [
    cls for cls in MODELWRAPPER_SUBCLASSES if not issubclass(cls, LLM)
]

# mock ai8x tools paths
if "AI8X_TRAINING_PATH" not in os.environ:
    os.environ["AI8X_TRAINING_PATH"] = "."

if "AI8X_SYNTHESIS_PATH" not in os.environ:
    os.environ["AI8X_SYNTHESIS_PATH"] = "."


@pytest.fixture(autouse=True, scope="module")
def prepare_models_io_specs():
    for model_cls in MODELWRAPPER_SUBCLASSES:
        if (
            model_cls.default_dataset is None
            or model_cls.pretrained_model_uri is None
        ):
            continue
        dataset_cls = model_cls.default_dataset
        dataset = get_dataset_random_mock(dataset_cls, model_cls)

        model_path = ResourceURI(model_cls.pretrained_model_uri)

        model = model_cls(model_path, dataset, from_file=True)
        model.save_io_specification(model_path)


def ensure_types_native_or_numpy(X):
    from typing import Dict, List, Tuple

    import numpy as np

    if isinstance(X, List) or isinstance(X, Tuple):
        for x in X:
            ensure_types_native_or_numpy(x)
    elif isinstance(X, Dict):
        for _, x in X.items():
            ensure_types_native_or_numpy(x)
    elif isinstance(X, np.ndarray):
        return
    elif isinstance(X, (int, float, str)):
        return
    else:
        raise TypeError(
            f"preprocess_input should return framework agnostic"
            f"values, got: { type(X) }"
        )


def create_model(model_cls: Type[ModelWrapper], dataset: Dataset):
    if model_cls.pretrained_model_uri is not None:
        model_path = copy_model_to_tmp(
            ResourceURI(model_cls.pretrained_model_uri)
        )
        from_file = True
    else:
        model_path = get_tmp_path()
        remove_file_or_dir(model_path)
        from_file = False
    return model_cls(model_path, dataset, from_file)


@pytest.fixture(scope="function")
def model(request):
    model_cls = request.param

    dataset_cls = model_cls.default_dataset
    dataset = get_dataset_random_mock(dataset_cls, model_cls)

    return create_model(model_cls, dataset)


class TestModelWrapper:
    @pytest.mark.parametrize(
        "model_cls",
        [
            pytest.param(
                cls,
                marks=[
                    pytest.mark.xdist_group(
                        name=f"TestModelWrapper_{cls.__name__}"
                    )
                ],
            )
            for cls in MODELWRAPPER_SUBCLASSES
        ],
    )
    def test_initializer_no_dataset(self, model_cls: Type[ModelWrapper]):
        """
        Tests model initialization without specified dataset.
        """
        _ = create_model(model_cls, None)

    @pytest.mark.parametrize(
        "model_cls",
        [
            pytest.param(
                cls,
                marks=[
                    pytest.mark.dependency(
                        name=f"test_initializer_with_dataset[{cls.__name__}]"
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestModelWrapper_{cls.__name__}"
                    ),
                ],
            )
            for cls in MODELWRAPPER_SUBCLASSES
        ],
    )
    def test_initializer_with_dataset(self, model_cls: Type[ModelWrapper]):
        """
        Tests model initialization with specified dataset.
        """
        dataset_cls = model_cls.default_dataset
        if dataset_cls is None:
            pytest.xfail("default dataset was not defined")
        dataset = get_dataset_random_mock(dataset_cls, model_cls)
        _ = create_model(model_cls, dataset)

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(
                cls,
                marks=[
                    pytest.mark.dependency(
                        name=f"test_prepare[{cls.__name__}]",
                        depends=[
                            f"test_initializer_with_dataset[{cls.__name__}]"
                        ],
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestModelWrapper_{cls.__name__}"
                    ),
                ],
            )
            for cls in MODELWRAPPER_SUBCLASSES
        ],
        indirect=True,
    )
    def test_prepare(self, model: Type[ModelWrapper]):
        """
        Tests the `prepare_model` method.
        """
        if isinstance(model, Ai8xAnomalyDetectionCNN):
            pytest.xfail("Ai8xAnomalyDetectionCNN requires ai8x-training")
        model.prepare_model()

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(
                cls,
                marks=[
                    pytest.mark.dependency(
                        depends=[f"test_prepare[{cls.__name__}]"]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestModelWrapper_{cls.__name__}"
                    ),
                ],
            )
            for cls in MODELWRAPPER_SUBCLASSES
        ],
        indirect=True,
    )
    def test_save(self, model: Type[ModelWrapper]):
        """
        Tests the `save_model` method.
        """
        model.prepare_model()
        model_save_path = get_tmp_path()
        try:
            model.save_model(model_save_path)
        except NotSupportedError:
            pytest.xfail("save_model not implemented for this model")
        assert model_save_path.exists()

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(
                cls,
                marks=[
                    pytest.mark.dependency(
                        depends=[f"test_prepare[{cls.__name__}]"]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestModelWrapper_{cls.__name__}"
                    ),
                ],
            )
            for cls in MODELWRAPPER_SUBCLASSES
        ],
        indirect=True,
    )
    def test_preprocess_types(self, model: Type[ModelWrapper]):
        model.prepare_model()

        sample, y = next(iter(model.dataset.iter_test()))
        prepX = model._preprocess_input(sample)

        ensure_types_native_or_numpy(prepX)

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(
                cls,
                marks=[
                    pytest.mark.dependency(
                        depends=[f"test_prepare[{cls.__name__}]"]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestModelWrapper_{cls.__name__}"
                    ),
                ],
            )
            for cls in MODELWRAPPER_SUBCLASSES
        ],
        indirect=True,
    )
    def test_inference(self, model: Type[ModelWrapper]):
        """
        Tests the `test_inference` method.
        """
        model.prepare_model()
        try:
            model.test_inference()
        except NotSupportedError:
            pytest.xfail("test_inference not implemented for this model")

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(
                cls,
                marks=[
                    pytest.mark.dependency(
                        depends=[
                            f"test_inference[{cls.__name__}]",
                            f"test_preprocess_types[{cls.__name__}]",
                        ]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestModelWrapper_{cls.__name__}"
                    ),
                ],
            )
            for cls in MODELWRAPPER_SUBCLASSES
        ],
        indirect=True,
    )
    def test_postprocess_types(self, model: Type[ModelWrapper]):
        model.prepare_model()
        sample, y = next(iter(model.dataset.iter_test()))
        prepX = model._preprocess_input(sample)
        preds = model._run_inference(prepX)
        ensure_types_native_or_numpy(preds)
        posty = model._postprocess_outputs(preds)
        ensure_types_native_or_numpy(posty)

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(
                cls,
                marks=[
                    pytest.mark.dependency(
                        depends=[f"test_prepare[{cls.__name__}]"]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestModelWrapper_{cls.__name__}"
                    ),
                ],
            )
            for cls in MODELWRAPPER_SUBCLASSES
        ],
        indirect=True,
    )
    def test_train(self, model: ModelWrapper):
        """
        Tests the `train_model` method.
        """
        model.prepare_model()

        model.batch_size = 16
        model.learning_rate = 0.01
        model.num_epochs = 1
        model.logdir = Path("/tmp/logdir")

        try:
            model.train_model()
        except NotSupportedError:
            pytest.xfail("train_model not implemented for this model")

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(
                cls,
                marks=[
                    pytest.mark.dependency(
                        depends=[f"test_prepare[{cls.__name__}]"]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestModelWrapper_{cls.__name__}"
                    ),
                ],
            )
            for cls in MODELWRAPPER_SUBCLASSES
        ],
        indirect=True,
    )
    def test_get_io_spec(self, model: Type[ModelWrapper]):
        """
        Tests the `train_model` method.
        """
        assert model.get_io_specification_from_model() is not None

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(
                cls,
                marks=[
                    pytest.mark.dependency(
                        depends=[f"test_prepare[{cls.__name__}]"]
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestModelWrapper_{cls.__name__}"
                    ),
                ],
            )
            for cls in MODELWRAPPER_SUBCLASSES
        ],
        indirect=True,
    )
    def test_get_framework_and_version(self, model: Type[ModelWrapper]):
        """
        Tests the `train_model` method.
        """
        assert model.get_framework_and_version() is not None

    def test_substitutive_method_of_loading_tf_model(self):
        """
        Test inference on a loaded model with the backup model
        loading method.
        """
        from kenning.modelwrappers.classification.tensorflow_imagenet import (
            TensorFlowImageNet,
        )

        if TensorFlowImageNet.default_dataset is None:
            raise ValueError("The dataset class is not available.")

        model_path = get_tmp_path(suffix=".keras")
        dataset = get_dataset_random_mock(TensorFlowImageNet.default_dataset)

        assert TensorFlowImageNet.pretrained_model_uri is not None
        ResourceManager().get_resource(
            TensorFlowImageNet.pretrained_model_uri, model_path
        )

        def mock_load_model(*args, **kwargs):
            raise ModelNotLoadedError(
                "This model loading method will always fail."
            )

        with patch("tensorflow.keras.models.load_model", mock_load_model):
            loaded_model = create_model(TensorFlowImageNet, dataset)
            loaded_model.load_model(model_path)
            loaded_model.test_inference()
