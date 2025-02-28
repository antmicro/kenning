# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from shutil import rmtree
from tempfile import mkdtemp
from typing import Callable, Optional, Type

import pytest

from kenning.core.automl import AutoML
from kenning.platforms.local import LocalPlatform
from kenning.tests.core.conftest import (
    get_dataset_random_mock,
)
from kenning.utils.class_loader import (
    MODEL_WRAPPERS,
    get_all_subclasses,
    load_class_by_type,
)
from kenning.utils.pipeline_runner import PipelineRunner

AUTOML_SUBCLASSES = get_all_subclasses(
    "kenning.automl", AutoML, raise_exception=True
)


def automl_matrix_test(
    param_name: str,
    depend: Optional[str] = None,
    indirect: bool = False,
) -> Callable:
    def inner_func(func: Callable) -> Callable:
        return pytest.mark.xdist_group(name="use_resources")(
            pytest.mark.parametrize(
                param_name,
                [
                    pytest.param(
                        cls,
                        marks=[
                            pytest.mark.dependency(
                                name=f"{func.__name__}[{cls.__name__}]",
                                depends=[f"{depend}[{cls.__name__}]"]
                                if depend
                                else [],
                            ),
                            pytest.mark.xdist_group(
                                name=f"TestAutoML_{cls.__name__}"
                            ),
                            pytest.mark.automl,
                        ],
                    )
                    for cls in AUTOML_SUBCLASSES
                ],
                indirect=indirect,
            )(func)
        )

    return inner_func


@pytest.fixture(scope="function")
def automl(request):
    automl_cls = request.param

    supported_models = automl_cls.supported_models
    if not supported_models:
        return None
    model_path = supported_models[0]
    model_cls = load_class_by_type(model_path, MODEL_WRAPPERS)
    dataset_cls = model_cls.default_dataset

    dataset = get_dataset_random_mock(dataset_cls, model_cls)
    tmp_dir = Path(mkdtemp())

    yield automl_cls(
        dataset=dataset,
        platform=LocalPlatform(),
        output_directory=tmp_dir,
    )

    rmtree(tmp_dir)


class TestAutoML:
    @automl_matrix_test("automl_cls")
    def test_initializer(self, automl_cls: Type[AutoML]):
        """
        Tests AutoML initialization.

        This should assert that specified `supported_models`
        inherits from AutoMLModel.
        """
        _ = automl_cls(None, None, Path("."))

    @pytest.mark.xfail(strict=True, raises=AssertionError)
    @automl_matrix_test("automl_cls")
    def test_initializer_with_unsupported_models(
        self, automl_cls: Type[AutoML]
    ):
        """
        Tests AutoML initialization.

        This should fail due to the provided model
        not implementing AutoMLModel.
        """
        _ = automl_cls(
            None,
            None,
            Path("."),
            use_models=[
                "kenning.modelwrappers.classification.pytorch_pet_dataset.PyTorchPetDatasetMobileNetV2"
            ],
        )

    @automl_matrix_test("automl", depend="test_initializer", indirect=True)
    def test_prepare_framework(self, automl: AutoML):
        """
        Test AutoML framework preparation.
        """
        print(automl)
        automl.prepare_framework()

    @automl_matrix_test(
        "automl",
        depend="test_prepare_framework",
        indirect=True,
    )
    def test_search(self, automl: AutoML):
        """
        Test AutoML search process and best configs.
        """
        automl.prepare_framework()
        automl.time_limit = 2.5

        automl.search()

        configs = automl.get_best_configs()
        assert (
            len(configs) <= automl.n_best_models
        ), f"Method should return at most {automl.n_best_models}"
        for config in configs:
            try:
                PipelineRunner.from_json_cfg(json_cfg=config)
            except Exception:
                pytest.fail(
                    f"Generated configuration ({config}) is not valid "
                    "and cannot initialize the pipeline",
                    pytrace=True,
                )
