# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from typing import Type
from pathlib import Path
import shutil
import numpy as np
import cv2

from kenning.core.dataset import Dataset
from kenning.core.dataset import CannotDownloadDatasetError
from kenning.utils.class_loader import get_all_subclasses
from kenning.tests.core.conftest import get_reduced_dataset_path
from kenning.tests.core.conftest import get_dataset_download_path


DATASET_SUBCLASSES = get_all_subclasses(
    "kenning.datasets", Dataset, raise_exception=True
)


@pytest.fixture(scope="function")
def dataset(request):
    dataset_cls = request.param

    path = path_reduced = get_reduced_dataset_path(dataset_cls)
    if not path.exists():
        path = get_dataset_download_path(dataset_cls)
    if not path.exists() and "Random" not in dataset_cls.__name__:
        pytest.xfail(
            f"Dataset {dataset_cls.__name__} not found in any of {path} and "
            f"{path_reduced} directories"
        )

    try:
        dataset = dataset_cls(path, download_dataset=False)
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
                ],
            )
            for dataset_cls in DATASET_SUBCLASSES
        ],
    )
    def test_folder_does_not_exist(self, dataset_cls: Type[Dataset]):
        """
        Tests throwing exception when there is no folder with data.
        """
        dataset_download_dir = get_dataset_download_path(dataset_cls)
        dataset_download_dir = dataset_download_dir.with_name(
            dataset_download_dir.name + "_none"
        )
        if dataset_download_dir.exists():
            shutil.rmtree(str(dataset_download_dir), ignore_errors=True)

        try:
            dataset = dataset_cls(dataset_download_dir, download_dataset=False)
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
        """
        for i, (x, y) in enumerate(dataset):
            assert x is not None
            assert y is not None
            if i > 10:
                break

        assert len(dataset) > 0

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
    def test_images_loading(self, dataset: Type[Dataset]):
        """
        Tests dataset iteration.
        """
        if "Random" in dataset.__class__.__name__:
            pytest.skip("random dataset does not load images")
        X_sample = dataset.dataX[0]

        if not (
            isinstance(X_sample, str)
            and (
                X_sample.split(".")[-1].lower() in ("jpg", "png", "jpeg")
                or hasattr(dataset, "get_sample_image_path")
            )
        ):
            pytest.skip("dataset inputs are not images")

        N = 10

        # disable images preprocessing
        dataset.standardize = False
        dataset.image_memory_layout = "NHWC"
        dataset.preprocess_type = None
        # generate random images
        sample_shape = dataset.prepare_input_samples([X_sample])[0].shape
        random_images = np.random.randint(
            0, 255, size=(N, 8, 8, sample_shape[2]), dtype=np.uint8
        )
        # assert that full range is used
        random_images[0, 0, 0, 0] = 0
        random_images[0, 0, 1, 0] = 255

        # write random images to dataset files
        random_images_resized = np.zeros((N, *sample_shape), dtype=np.uint8)

        for i in range(N):
            resized_img = cv2.resize(
                random_images[i],
                (sample_shape[1], sample_shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            if resized_img.ndim == 2:
                resized_img = np.expand_dims(resized_img, -1)

            random_images_resized[i] = resized_img
            img_path = dataset.dataX[i]
            if hasattr(dataset, "get_sample_image_path"):
                img_path = dataset.get_sample_image_path(img_path)
            if not Path(img_path).exists():
                raise FileNotFoundError

            cv2.imwrite(img_path, random_images_resized[i])

        # load images by dataset
        loaded_images = dataset.prepare_input_samples(dataset.dataX[:N])
        loaded_images = np.array(loaded_images)

        # convert to the same range
        random_images_resized = (
            random_images_resized.astype(np.float32) / 255.0
        )
        if np.max(loaded_images) > 1.0:
            loaded_images /= 255.0

        # compare shapes
        assert random_images_resized.shape == loaded_images.shape
        # compare similarity (RGB or BGR as color format cannot be retrieved
        # from dataset)
        assert (
            np.mean(np.abs(loaded_images - random_images_resized)) < 0.05
            or np.mean(
                np.abs(loaded_images - random_images_resized[:, :, :, ::-1])
            )
            < 0.05
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
                ],
            )
            for dataset_cls in DATASET_SUBCLASSES
        ],
    )
    def test_dataset_checksum_verification(self, dataset_cls: Type[Dataset]):
        """
        Tests if dataset checksum is properly calculated.
        """
        if "Random" in dataset_cls.__name__:
            pytest.skip("random dataset does not have files")

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
                ],
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
        if "Random" in dataset_cls.__name__:
            pytest.skip("random dataset does not have files")

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
            dataset_path, download_dataset=True, force_download_dataset=False
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
                ],
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
        if "Random" in dataset_cls.__name__:
            pytest.skip("random dataset does not have files")

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
            dataset_path, download_dataset=True, force_download_dataset=False
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
                ],
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
        if "Random" in dataset_cls.__name__:
            pytest.skip("random dataset does not have files")

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
            dataset_path, download_dataset=True, force_download_dataset=True
        )

        assert mock_download_dataset_fun.num_calls == 1

        mock_download_dataset_fun.num_calls = 0
        dataset_cls.verify_dataset_checksum = lambda self: True

        dataset_cls(
            dataset_path, download_dataset=True, force_download_dataset=True
        )

        assert mock_download_dataset_fun.num_calls == 1
