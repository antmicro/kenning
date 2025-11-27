# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import shutil
from math import ceil
from pathlib import Path
from typing import Type

import pytest

from kenning.core.dataset import Dataset
from kenning.core.exceptions import CannotDownloadDatasetError
from kenning.tests.core.conftest import (
    get_dataset_download_path,
    get_reduced_dataset_path,
)
from kenning.utils.class_loader import get_all_subclasses

DATASET_SUBCLASSES = get_all_subclasses(
    "kenning.datasets", Dataset, raise_exception=True
)

NOT_DOWNLOADED_DATASETS = ["Lindenthal", "VideoDataset"]


@pytest.fixture(scope="function")
def dataset(request):
    dataset_cls = request.param

    path = path_reduced = get_reduced_dataset_path(dataset_cls)

    kwargs = {}
    if not path.exists():
        path = get_dataset_download_path(dataset_cls)
    if not path.exists() and "Random" not in dataset_cls.__name__:
        pytest.xfail(
            f"Dataset {dataset_cls.__name__} not found in any of {path} and "
            f"{path_reduced} directories"
        )
    if "TabularDataset" in dataset_cls.__name__:
        kwargs["dataset_path"] = "asdfg"
        kwargs["colsX"] = ["x1", "x2"]
        kwargs["colY"] = "y"

    try:
        dataset = dataset_cls(
            path,
            download_dataset=False,
            **kwargs,
        )
        assert len(dataset.dataX) > 0
    except CannotDownloadDatasetError:
        pytest.xfail("Cannot download dataset.")
    except Exception as e:
        pytest.fail(f"Exception {e}")
    return dataset


class TestDataset:
    @pytest.mark.parametrize(
        "dataset_cls",
        [
            pytest.param(
                dataset_cls,
                marks=[
                    pytest.mark.xdist_group(
                        name=f"TestDataset_{dataset_cls.__name__}"
                    )
                ]
                + (
                    [pytest.mark.skip("Method not implemented")]
                    if dataset_cls.__name__
                    == "VideoObjectDetectionSegmentationDataset"
                    else []
                ),
            )
            for dataset_cls in DATASET_SUBCLASSES
        ],
    )
    def test_folder_does_not_exist(self, dataset_cls: Type[Dataset]):
        """
        Tests throwing exception when there is no folder with data.
        """
        kwargs = {}
        if "TabularDataset" in dataset_cls.__name__:
            kwargs["dataset_path"] = "asdfg"
            kwargs["colsX"] = ["x1", "x2"]
            kwargs["colY"] = "y"

        dataset_download_dir = get_dataset_download_path(dataset_cls)
        dataset_download_dir = dataset_download_dir.with_name(
            dataset_download_dir.name + "_none"
        )
        if dataset_download_dir.exists():
            shutil.rmtree(str(dataset_download_dir), ignore_errors=True)

        try:
            dataset = dataset_cls(
                dataset_download_dir,
                download_dataset=False,
                **kwargs,
            )
            dataset.prepare()
            if "Random" not in dataset_cls.__name__:
                pytest.fail("No exception thrown")
        except FileNotFoundError:
            pass
        except Exception as e:
            pytest.fail(f"Exception {e}")

    @pytest.mark.skip(reason="avoiding hitting download rate limit")
    @pytest.mark.parametrize(
        "dataset_cls",
        [
            pytest.param(
                dataset_cls,
                marks=[
                    pytest.mark.dependency(
                        name=f"test_download[{dataset_cls.__name__}]"
                    ),
                    pytest.mark.xdist_group(
                        name=f"TestDataset_{dataset_cls.__name__}"
                    ),
                ],
            )
            for dataset_cls in DATASET_SUBCLASSES
        ],
    )
    def test_download(self, dataset_cls: Type[Dataset]):
        """
        Tests dataset downloading.
        """
        dataset_download_dir = get_dataset_download_path(dataset_cls)
        if dataset_download_dir.exists():
            shutil.rmtree(dataset_download_dir, ignore_errors=True)

        try:
            dataset = dataset_cls(dataset_download_dir, download_dataset=True)
            assert len(dataset.dataX) > 0
        except CannotDownloadDatasetError:
            pytest.xfail("Cannot download dataset.")
        except Exception as e:
            pytest.fail(f"Exception {e}")

    @pytest.mark.parametrize(
        "dataset",
        [
            pytest.param(
                dataset_cls,
                marks=[
                    pytest.mark.xdist_group(
                        name=f"TestDataset_{dataset_cls.__name__}"
                    )
                ],
            )
            for dataset_cls in DATASET_SUBCLASSES
        ],
        indirect=True,
    )
    def test_iterator(self, dataset: Type[Dataset]):
        """
        Tests dataset iteration.
        Verifies that indexes are generated and used.
        """
        for i, (x, y) in enumerate(dataset):
            assert x is not None
            assert y is not None
            if i > 10:
                break

        assert len(dataset) > 0
        assert len(dataset._dataindices) > 0
        assert dataset._dataindex == (i + 1) * dataset.batch_size

    @pytest.mark.parametrize(
        "dataset",
        [
            pytest.param(
                dataset_cls,
                marks=[
                    pytest.mark.xdist_group(
                        name=f"TestDataset_{dataset_cls.__name__}"
                    )
                ],
            )
            for dataset_cls in DATASET_SUBCLASSES
        ],
        indirect=True,
    )
    def test_iterator_last_batch(self, dataset: Type[Dataset]):
        """
        Tests dataset's internal index after the last batch with different
        length than the batch size.
        Verifies that indexes are tracked correctly.
        """
        assert len(dataset) > 0
        assert len(dataset._dataindices) == 0
        for _ in dataset:
            break
        dataset.batch_size = 10
        dataset._dataindex = dataset._dataindices[-3]
        x, y = dataset.__next__()
        assert x is not None
        assert y is not None
        assert len(x[0]) == len(y[0]) == 3
        assert dataset._dataindex == len(dataset.dataX)
        with pytest.raises(StopIteration):
            dataset.__next__()

    @pytest.mark.parametrize(
        "dataset",
        [
            pytest.param(
                dataset_cls,
                marks=[
                    pytest.mark.xdist_group(
                        name=f"TestDataset_{dataset_cls.__name__}"
                    )
                ],
            )
            for dataset_cls in DATASET_SUBCLASSES
        ],
        indirect=True,
    )
    def test_iterator_len(self, dataset: Type[Dataset]):
        """
        Tests length of dataset iteration.
        Verifies whether batch size is taken into account.
        """
        iter_train = dataset.iter_test()
        split = dataset.split_fraction_test

        assert len(iter_train) == ceil(
            len(dataset.dataX) * split / dataset.batch_size
        )

    @pytest.mark.parametrize(
        "dataset",
        [
            pytest.param(
                dataset_cls,
                marks=[
                    pytest.mark.xdist_group(
                        name=f"TestDataset_{dataset_cls.__name__}"
                    )
                ],
            )
            for dataset_cls in DATASET_SUBCLASSES
        ],
        indirect=True,
    )
    def test_data_equal_length(self, dataset: Type[Dataset]):
        """
        Tests dataset iteration.
        """
        assert len(dataset) == len(dataset.dataX)
        assert len(dataset.dataX) == len(dataset.dataY)

    @pytest.mark.parametrize(
        "dataset",
        [
            pytest.param(
                dataset_cls,
                marks=[
                    pytest.mark.xdist_group(
                        name=f"TestDataset_{dataset_cls.__name__}"
                    )
                ],
            )
            for dataset_cls in DATASET_SUBCLASSES
        ],
        indirect=True,
    )
    def test_train_test_split(self, dataset: Type[Dataset]):
        """
        Tests the `train_test_split_representations` method.
        """
        test_fraction = 0.25
        (
            dataXtrain,
            dataXtest,
            dataYtrain,
            dataYtest,
        ) = dataset.train_test_split_representations(test_fraction)

        assert len(dataXtrain) > 0
        assert len(dataXtest) > 0
        assert len(dataYtrain) > 0
        assert len(dataYtest) > 0
        assert len(dataXtrain) + len(dataXtest) == len(dataset.dataX)
        assert len(dataYtrain) + len(dataYtest) == len(dataset.dataY)

        tolerance = 1.0 / len(dataset)

        assert len(dataXtrain) / len(dataset.dataX) == pytest.approx(
            1 - test_fraction, abs=tolerance
        )
        assert len(dataYtrain) / len(dataset.dataY) == pytest.approx(
            1 - test_fraction, abs=tolerance
        )
        assert len(dataXtest) / len(dataset.dataX) == pytest.approx(
            test_fraction, abs=tolerance
        )
        assert len(dataYtest) / len(dataset.dataY) == pytest.approx(
            test_fraction, abs=tolerance
        )

    @pytest.mark.parametrize(
        "dataset",
        [
            pytest.param(
                dataset_cls,
                marks=[
                    pytest.mark.xdist_group(
                        name=f"TestDataset_{dataset_cls.__name__}"
                    )
                ],
            )
            for dataset_cls in DATASET_SUBCLASSES
        ],
        indirect=True,
    )
    def test_train_test_val_split(self, dataset: Type[Dataset]):
        """
        Tests the `train_test_split_representations` method.
        """
        test_fraction = 0.2
        val_fraction = 0.2
        (
            dataXtrain,
            dataXtest,
            dataYtrain,
            dataYtest,
            dataXval,
            dataYval,
        ) = dataset.train_test_split_representations(
            test_fraction, val_fraction=val_fraction
        )

        assert len(dataXtrain) > 0
        assert len(dataYtrain) > 0
        assert len(dataXtest) > 0
        assert len(dataYtest) > 0
        assert len(dataXval) > 0
        assert len(dataYval) > 0
        assert len(dataXtrain) + len(dataXtest) + len(dataXval) == len(
            dataset.dataX
        )
        assert len(dataYtrain) + len(dataYtest) + len(dataYval) == len(
            dataset.dataY
        )

        tolerance = 2.0 / len(dataset)
        train_fraction = 1 - test_fraction - val_fraction

        assert len(dataXtrain) / len(dataset.dataX) == pytest.approx(
            train_fraction, abs=tolerance
        )
        assert len(dataYtrain) / len(dataset.dataY) == pytest.approx(
            train_fraction, abs=tolerance
        )
        assert len(dataXtest) / len(dataset.dataX) == pytest.approx(
            test_fraction, abs=tolerance
        )
        assert len(dataYtest) / len(dataset.dataY) == pytest.approx(
            test_fraction, abs=tolerance
        )
        assert len(dataXval) / len(dataset.dataY) == pytest.approx(
            val_fraction, abs=tolerance
        )
        assert len(dataYval) / len(dataset.dataY) == pytest.approx(
            val_fraction, abs=tolerance
        )

    @pytest.mark.parametrize(
        "dataset_cls",
        [
            pytest.param(
                dataset_cls,
                marks=[
                    pytest.mark.xdist_group(
                        name=f"TestDataset_{dataset_cls.__name__}"
                    )
                ]
                + (
                    [pytest.mark.skip("Method not implemented")]
                    if dataset_cls.__name__
                    == "VideoObjectDetectionSegmentationDataset"
                    else []
                ),
            )
            for dataset_cls in DATASET_SUBCLASSES
        ],
    )
    def test_dataset_checksum_verification(self, dataset_cls: Type[Dataset]):
        """
        Tests if dataset checksum is properly calculated.
        """
        kwargs = {}
        if "Random" in dataset_cls.__name__:
            pytest.skip("random dataset does not have files")
        elif "Lindenthal" in dataset_cls.__name__:
            pytest.xfail("Lindenthal dataset does not have files downloaded")
        if "VideoDataset" in dataset_cls.__name__:
            pytest.skip("Video dataset does not have any files")
        elif "TabularDataset" in dataset_cls.__name__:
            kwargs["dataset_path"] = "asdfg"
            kwargs["colsX"] = ["x1", "x2"]
            kwargs["colY"] = "y"

        dataset_path = get_reduced_dataset_path(dataset_cls)
        dataset_path = dataset_path.with_name(dataset_path.name + "_test")

        def mock_download_dataset_fun(self):
            shutil.copytree(
                get_reduced_dataset_path(dataset_cls), dataset_path
            )

        dataset_cls.download_dataset_fun = mock_download_dataset_fun
        dataset = dataset_cls(
            root=dataset_path,
            download_dataset=True,
            force_download_dataset=False,
            **kwargs,
        )

        new_file = Path(dataset.root / "some_new_file.txt")
        if new_file.is_file():
            new_file.unlink()

        checksum_file = Path(dataset.root / "DATASET_CHECKSUM")
        if checksum_file.is_file():
            checksum_file.unlink()

        # check if checksum is properly verified
        dataset.save_dataset_checksum()

        assert checksum_file.is_file()
        assert dataset.verify_dataset_checksum()

        # check if verify method returns false when there are some changes in
        # dataset directory
        new_file.write_text("some_data")

        assert checksum_file.is_file()
        assert not dataset.verify_dataset_checksum()

        # check if checksum is properly updated after changes
        dataset.save_dataset_checksum()

        assert checksum_file.is_file()
        assert dataset.verify_dataset_checksum()

        # check if verify return false when there is no checksum file
        checksum_file.unlink()

        assert not checksum_file.exists()
        assert not dataset.verify_dataset_checksum()

        new_file.unlink()

    @pytest.mark.parametrize(
        "dataset_cls",
        [
            pytest.param(
                dataset_cls,
                marks=[
                    pytest.mark.xdist_group(
                        name=f"TestDataset_{dataset_cls.__name__}"
                    )
                ]
                + (
                    [pytest.mark.skip("Method not implemented")]
                    if dataset_cls.__name__
                    == "VideoObjectDetectionSegmentationDataset"
                    else []
                ),
            )
            for dataset_cls in DATASET_SUBCLASSES
        ],
    )
    def test_if_dataset_is_not_downloaded_when_checksum_is_valid(
        self, dataset_cls: Type[Dataset]
    ):
        """
        Tests if download is skipped when dataset files are already downloaded.
        """
        kwargs = {}
        if "Random" in dataset_cls.__name__:
            pytest.skip("random dataset does not have files")
        if "Lindenthal" in dataset_cls.__name__:
            pytest.xfail("Lindenthal dataset does not have files downloaded")
        if "VideoDataset" in dataset_cls.__name__:
            pytest.skip("Video dataset does not have any files")
        elif "TabularDataset" in dataset_cls.__name__:
            kwargs["dataset_path"] = "asdfg"
            kwargs["colsX"] = ["x1", "x2"]
            kwargs["colY"] = "y"

        dataset_path = get_reduced_dataset_path(dataset_cls)
        dataset_path = dataset_path.with_name(dataset_path.name + "_test")

        def mock_download_dataset_fun(self):
            shutil.copytree(
                get_reduced_dataset_path(dataset_cls), dataset_path
            )
            mock_download_dataset_fun.num_calls += 1

        mock_download_dataset_fun.num_calls = 0
        dataset_cls.download_dataset_fun = mock_download_dataset_fun
        dataset_cls.verify_dataset_checksum = lambda self: True

        dataset_cls(
            dataset_path,
            download_dataset=True,
            force_download_dataset=False,
            **kwargs,
        )

        assert mock_download_dataset_fun.num_calls == 0

    @pytest.mark.parametrize(
        "dataset_cls",
        [
            pytest.param(
                dataset_cls,
                marks=[
                    pytest.mark.xdist_group(
                        name=f"TestDataset_{dataset_cls.__name__}"
                    )
                ]
                + (
                    [pytest.mark.skip("Method not implemented")]
                    if dataset_cls.__name__
                    == "VideoObjectDetectionSegmentationDataset"
                    else []
                ),
            )
            for dataset_cls in DATASET_SUBCLASSES
        ],
    )
    def test_if_dataset_is_downloaded_when_checksum_is_invalid(
        self, dataset_cls: Type[Dataset]
    ):
        """
        Tests if dataset is downloaded when dataset checksum is invalid.
        """
        kwargs = {}
        if "Random" in dataset_cls.__name__:
            pytest.skip("random dataset does not have files")
        if "Lindenthal" in dataset_cls.__name__:
            pytest.xfail("Lindenthal dataset does not have files downloaded")
        if "VideoDataset" in dataset_cls.__name__:
            pytest.skip("Video dataset does not have any files")
        elif "TabularDataset" in dataset_cls.__name__:
            kwargs["dataset_path"] = "asdfg"
            kwargs["colsX"] = ["x1", "x2"]
            kwargs["colY"] = "y"

        dataset_path = get_reduced_dataset_path(dataset_cls)
        dataset_path = dataset_path.with_name(dataset_path.name + "_test")

        def mock_download_dataset_fun(self):
            shutil.copytree(
                get_reduced_dataset_path(dataset_cls), dataset_path
            )
            mock_download_dataset_fun.num_calls += 1

        mock_download_dataset_fun.num_calls = 0
        dataset_cls.download_dataset_fun = mock_download_dataset_fun
        dataset_cls.verify_dataset_checksum = lambda self: False

        dataset_cls(
            dataset_path,
            download_dataset=True,
            force_download_dataset=False,
            **kwargs,
        )

        assert mock_download_dataset_fun.num_calls == 1

    @pytest.mark.parametrize(
        "dataset_cls",
        [
            pytest.param(
                dataset_cls,
                marks=[
                    pytest.mark.xdist_group(
                        name=f"TestDataset_{dataset_cls.__name__}"
                    )
                ]
                + (
                    [pytest.mark.skip("Method not implemented")]
                    if dataset_cls.__name__
                    == "VideoObjectDetectionSegmentationDataset"
                    else []
                ),
            )
            for dataset_cls in DATASET_SUBCLASSES
        ],
    )
    def test_if_dataset_is_downloaded_when_force_download_is_true(
        self, dataset_cls: Type[Dataset]
    ):
        """
        Tests if dataset is downloaded when dataset checksum is invalid.
        """
        kwargs = {}
        if "Random" in dataset_cls.__name__:
            pytest.skip("random dataset does not have files")
        if "Lindenthal" in dataset_cls.__name__:
            pytest.xfail("Lindenthal dataset does not have files downloaded")
        if "VideoDataset" in dataset_cls.__name__:
            pytest.skip("Video dataset does not have any files")
        elif "TabularDataset" in dataset_cls.__name__:
            kwargs["dataset_path"] = "asdfg"
            kwargs["colsX"] = ["x1", "x2"]
            kwargs["colY"] = "y"

        dataset_path = get_reduced_dataset_path(dataset_cls)
        dataset_path = dataset_path.with_name(dataset_path.name + "_test")

        def mock_download_dataset_fun(self):
            shutil.copytree(
                get_reduced_dataset_path(dataset_cls), dataset_path
            )
            mock_download_dataset_fun.num_calls += 1

        mock_download_dataset_fun.num_calls = 0
        dataset_cls.download_dataset_fun = mock_download_dataset_fun
        dataset_cls.verify_dataset_checksum = lambda self: False

        dataset_cls(
            dataset_path,
            download_dataset=True,
            force_download_dataset=True,
            **kwargs,
        )

        assert mock_download_dataset_fun.num_calls == 1

        mock_download_dataset_fun.num_calls = 0
        dataset_cls.verify_dataset_checksum = lambda self: True

        dataset_cls(
            dataset_path,
            download_dataset=True,
            force_download_dataset=True,
            **kwargs,
        )

        assert mock_download_dataset_fun.num_calls == 1
