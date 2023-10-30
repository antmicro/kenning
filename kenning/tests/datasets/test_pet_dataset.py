# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

import string
import tempfile
from pathlib import Path
from random import choices, randint

import pytest

from kenning.datasets.pet_dataset import PetDataset
from kenning.tests.conftest import DataFolder


def random_string() -> str:
    """
    Generates random string with length from 0 till 1000.

    Returns
    -------
    str: Random string out of string.printables
    """
    return "".join(choices(string.printable, k=randint(0, 1000)))


@pytest.fixture()
def annotations() -> Path:
    """
    Creates temporary folder for PetDataset dataset.

    Returns
    -------
    Path: The path to temporary folder.
    """
    path = Path(tempfile.NamedTemporaryFile().name)
    path.mkdir()
    (path / "annotations").mkdir()
    (path / "annotations" / "list.txt").touch()
    yield path
    import shutil

    shutil.rmtree(path)


@pytest.mark.fast
class TestPetDataset:
    def test_not_exist(self):
        """
        Tests dataset behaviour if incorrect arguments are passed
        to constructor.

        List of methods that are being tested
        --------------------------------
        PetDataset.__init__()
        """
        path = Path(tempfile.NamedTemporaryFile().name)

        with pytest.raises(FileNotFoundError) as execinfo:
            PetDataset(path, download_dataset=False)
        assert str(path / "annotations" / "list.txt") in str(execinfo.value)

        with pytest.raises(AssertionError):
            PetDataset(
                path,
                standardize=False,
                classify_by=random_string(),
                download_dataset=False,
            )

        with pytest.raises(AssertionError):
            PetDataset(
                path,
                standardize=False,
                image_memory_layout=random_string(),
                download_dataset=False,
            )

    def test_empty_folder(self, annotations: Path):
        """
        Tests dataset behaviour if empty 'annotations/list.txt' is passed
        as argument to constructor.

        List of methods that are being tested
        --------------------------------
        dataset.get()
        dataset.__next__()

        Used fixtures
        -------------
        annotations - to get folder with empty 'annotations/list.txt'
        """
        dataset = PetDataset(annotations, download_dataset=False)
        assert ([], []) == dataset.get_data()

        with pytest.raises(StopIteration):
            dataset.__next__()

    def test_incorrect_data(self, annotations: Path):
        """
        Tests dataset behaviour if incorrect data is passed to
        'annotations/list.txt'.

        List of methods that are being tested
        --------------------------------
        PetDataset.prepare()
        PetDataset.get_data()

        Used fixtures
        -------------
        annotations - to get folder with empty 'annotations/list.txt'
        """
        with open(annotations / "annotations" / "list.txt", "w") as f:
            [print(f"{random_string()}", file=f) for i in range(100)]

        with pytest.raises((IndexError, ValueError)) as execinfo:
            PetDataset(annotations, download_dataset=False)

        with open(annotations / "annotations" / "list.txt", "w") as f:
            [print(f"image_{i} 1 1 1", file=f) for i in range(100)]

        dataset = PetDataset(annotations, download_dataset=False)
        with pytest.raises(FileNotFoundError) as execinfo:
            dataset.get_data()
        assert "image_0" in str(execinfo.value)

    def test_correct_data(self, datasetimages: DataFolder):
        """
        Tests dataset behaviour if wrong parameters and correct data
        are passed.

        List of methods that are being tested
        --------------------------------
        PetDataset.__init__()
        PetDataset.__next__()

        Used fixtures
        -------------
        datasetimages - to get folder with correct data.
        """
        with pytest.raises(AssertionError):
            PetDataset(
                datasetimages.path, batch_size=-1, download_dataset=False
            )
        with pytest.raises(AssertionError):
            PetDataset(
                datasetimages.path, batch_size=0, download_dataset=False
            )
        dataset = PetDataset(
            datasetimages.path,
            batch_size=1,
            standardize=False,
            download_dataset=False,
        )
        data = dataset.__next__()
        assert isinstance(data, tuple)
        assert len(data) > 0 and isinstance(data[0], list)
