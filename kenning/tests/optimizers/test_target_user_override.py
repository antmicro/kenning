# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import List, Optional, Set

import pytest

from kenning.core.dataset import Dataset
from kenning.datasets.magic_wand_dataset import MagicWandDataset
from kenning.optimizers.iree import IREECompiler
from kenning.optimizers.tvm import TVMCompiler
from kenning.platforms.cuda import CUDAPlatform
from kenning.platforms.zephyr import ZephyrPlatform
from kenning.tests.core.conftest import get_dataset_random_mock
from kenning.utils.compiler_flag import CompilerFlag


@pytest.fixture
def dataset() -> Dataset:
    return get_dataset_random_mock(MagicWandDataset)


def assert_compiler_flags(
    additional_flags: List[CompilerFlag],
    override_flag: Optional[CompilerFlag],
    original_flags: List[CompilerFlag],
    actual_flags: List[CompilerFlag],
) -> None:
    """
    Ensure that the compiler flags that we have gotten
    are the ones that we expect. Panics when they are
    different.

    Parameters
    ----------
    additional_flags: List[CompilerFlag]
        The flags that do not exist in the original flags.

    override_flag: Optional[CompilerFlag]
        The flag that already exist and we want to override.

    original_flags: List[CompilerFlag]
        The original set of compiler flags.

    actual_flags: List[CompilerFlag]
        The resulting set of flags after processing.
    """

    def flags2set(flags: List[CompilerFlag]) -> Set[str]:
        return {str(flag) for flag in flags}

    if override_flag:
        expected = (
            {
                str(flag)
                for flag in original_flags
                if not flag.is_same_flag(override_flag)
            }
            | flags2set(additional_flags)
            | {str(override_flag)}
        )
    else:
        expected = flags2set(original_flags) | flags2set(additional_flags)

    actual = flags2set(actual_flags)

    assert (
        actual == expected
    ), "User-defined flags were overridden by platform flags"


class TestTargetUserOverride:
    def test_user_override_tvm(
        self,
        dataset: Dataset,
        tmpfolder: Path,
    ) -> None:
        machine = "max32690evkit/max32690/m4"
        platform = ZephyrPlatform(machine)

        compiled_model_path = tmpfolder / "model-mock.tar"

        # Non-existent attrs from the platform that we want to add
        # extra_attrs = "-fast-math-contract -fast-math-ninf"
        extra_attrs = "-constants-byte-alignment=4"
        # Existing attr that we want to replace
        override_attr = "-mcpu=cortex-m4-r"

        compiler = TVMCompiler(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            model_framework="tflite",
            target_attrs=f"{extra_attrs} {override_attr}",
        )

        compiler.read_platform(platform)
        compiler.init()

        extra_attrs = map(CompilerFlag, extra_attrs.split())
        original_flags = map(CompilerFlag, compiler.platform_target_attrs)
        override_flag = CompilerFlag(override_attr)
        actual_flags = compiler._get_target_attrs(tostring=False)
        assert_compiler_flags(
            extra_attrs, override_flag, original_flags, actual_flags
        )

    def test_user_no_override_tvm(
        self,
        dataset: Dataset,
        tmpfolder: Path,
    ) -> None:
        machine = "max32690evkit/max32690/m4"
        platform = ZephyrPlatform(machine)

        compiled_model_path = tmpfolder / "model-mock.tar"

        compiler = TVMCompiler(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            model_framework="tflite",
        )

        compiler.read_platform(platform)
        compiler.init()

        original_flags = map(CompilerFlag, compiler.platform_target_attrs)
        actual_flags = compiler._get_target_attrs(tostring=False)

        assert (
            compiler._get_target() == "zephyr"
        ), "The target was not set to platform default"

        assert_compiler_flags([], None, original_flags, actual_flags)

    def test_user_override_tvm_cuda(
        self, dataset: Dataset, tmpfolder: Path
    ) -> None:
        machine = "max32690evkit/max32690/m4"
        platform = CUDAPlatform(machine)

        compiled_model_path = tmpfolder / "model-mock.tar"

        compiler = TVMCompiler(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            model_framework="tflite",
            target="cuda",
        )

        compiler.read_platform(platform)
        compiler.init()

        assert (
            compiler._get_target() == "cuda"
        ), "User-supplied CUDA was overwritten"

    def test_no_platform_tvm(self, dataset: Dataset, tmpfolder: Path) -> None:
        compiled_model_path = tmpfolder / "model-mock.tar"

        compiler = TVMCompiler(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            model_framework="tflite",
        )

        compiler.init()

        assert compiler._get_target() == "llvm"
        assert compiler._get_target_attrs() == ""

    def test_user_override_iree(
        self, dataset: Dataset, tmpfolder: Path
    ) -> None:
        machine = "nvidia_rtx_4090"
        platform = CUDAPlatform(machine)

        compiled_model_path = tmpfolder / "model-mock.tar"

        compiler = IREECompiler(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            model_framework="tflite",
            backend="vulkan",
        )

        compiler.read_platform(platform)
        compiler.init()

        assert (
            compiler._get_backend() == "vulkan"
        ), "The backend was overridden"

    def test_user_no_override_iree(
        self, dataset: Dataset, tmpfolder: Path
    ) -> None:
        machine = "nvidia_rtx_4090"
        platform = CUDAPlatform(machine)

        compiled_model_path = tmpfolder / "model-mock.tar"

        compiler = IREECompiler(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            model_framework="tflite",
        )

        compiler.read_platform(platform)
        compiler.init()

        assert (
            compiler._get_backend() == "cuda"
        ), "The backend is not the default"

    def test_no_platform_iree(self, dataset: Dataset, tmpfolder: Path) -> None:
        compiled_model_path = tmpfolder / "model-mock.tar"

        compiler = IREECompiler(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            model_framework="tflite",
        )

        compiler.init()

        assert compiler._get_backend() == "vmvx"

    def test_empty_target_attrs_tvm(
        self, dataset: Dataset, tmpfolder: Path
    ) -> None:
        machine = "max32690evkit/max32690/m4"
        platform = ZephyrPlatform(machine)

        compiled_model_path = tmpfolder / "model-mock.tar"

        compiler = TVMCompiler(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            model_framework="tflite",
            target_attrs="",
        )

        compiler.read_platform(platform)
        compiler.init()

        original_flags = map(CompilerFlag, compiler.platform_target_attrs)
        actual_flags = compiler._get_target_attrs(tostring=False)

        assert_compiler_flags([], None, original_flags, actual_flags)

    def test_get_target_attrs_handles_none_platform_attrs(
        self, dataset: Dataset, tmpfolder: Path
    ) -> None:
        compiled_model_path = tmpfolder / "model-mock.tar"

        compiler = TVMCompiler(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            model_framework="tflite",
        )

        compiler.init()
        attrs = compiler._get_target_attrs(tostring=False)
        assert isinstance(attrs, list)

        platform = ZephyrPlatform("max32690evkit/max32690/m4")
        compiler = TVMCompiler(
            dataset=dataset,
            compiled_model_path=compiled_model_path,
            model_framework="tflite",
        )

        platform.target_attrs = None
        compiler.read_platform(platform)
        compiler.init()
        # should not raise and should return a safe empty value
        attrs = compiler._get_target_attrs(tostring=False)
        assert isinstance(attrs, list)
