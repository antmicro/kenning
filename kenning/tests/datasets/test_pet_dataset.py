import pytest
import tempfile
import string
from kenning.datasets.pet_dataset import PetDataset
from pathlib import Path
from random import choices, randint


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
    (path / 'annotations').mkdir()
    (path / 'annotations' / 'list.txt').touch()
    yield path
    import shutil
    shutil.rmtree(path)


class TestPetDataset:
    def test_not_exist(self):
        """
        Tests dataset behaviour if incorrect arguments are passed
        to constructor.

        List of methods are being tested
        --------------------------------
        PetDataset.__init__()
        """
        path = Path(tempfile.NamedTemporaryFile().name)

        with pytest.raises(FileNotFoundError) as execinfo:
            PetDataset(path)
        assert str(path / 'annotations' / 'list.txt') in str(execinfo.value)

        with pytest.raises(AssertionError):
            PetDataset(path, standardize=False,
                       classify_by=random_string())

        with pytest.raises(AssertionError):
            PetDataset(path, standardize=False,
                       image_memory_layout=random_string())

    def test_empty_folder(self, annotations):
        """
        Tests dataset behaviour if empty 'annotations/list.txt' is passed
        as argument to constructor.

        List of methods are being tested
        --------------------------------
        dataset.get()
        dataset.__next__()

        Used fixtures
        -------------
        annotations - to get folder with empty 'annotations/list.txt'
        """
        dataset = PetDataset(annotations)
        assert ([], []) == dataset.get_data()

        with pytest.raises(StopIteration):
            dataset.__next__()

    def test_incorrect_data(self, annotations):
        """
        Tests dataset behaviour if incorrect data is passed to
        'annotations/list.txt'.

        List of methods are being tested
        --------------------------------
        PetDataset.prepare()
        PetDataset.get_data()

        Used fixtures
        -------------
        annotations - to get folder with empty 'annotations/list.txt'
        """
        with open(annotations / 'annotations' / 'list.txt', 'w') as f:
            [print(f'{random_string()}', file=f) for i in range(100)]

        with pytest.raises((IndexError, ValueError)) as execinfo:
            PetDataset(annotations)

        with open(annotations / 'annotations' / 'list.txt', 'w') as f:
            [print(f'image_{i} 1 1 1', file=f) for i in range(100)]

        dataset = PetDataset(annotations)
        with pytest.raises(FileNotFoundError) as execinfo:
            dataset.get_data()
        assert 'image_0' in str(execinfo.value)

    def test_correct_data(self, fake_images):
        """
        Tests dataset behaviour if wrong parameters and correct data
        are passed.

        List of methods are being tested
        --------------------------------
        PetDataset.__init__()
        PetDataset.__next__()

        Used fixtures
        -------------
        fake_images - to get folder with correct data.
        """
        with pytest.raises(AssertionError):
            PetDataset(fake_images.path, batch_size=-1)
        with pytest.raises(AssertionError):
            PetDataset(fake_images.path, batch_size=0)
        dataset = PetDataset(fake_images.path, batch_size=1, standardize=False)
        data = dataset.__next__()
        assert isinstance(data, tuple)
        assert len(data) > 0 and isinstance(data[0], list)
