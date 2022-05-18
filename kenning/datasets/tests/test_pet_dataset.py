from kenning.datasets.pet_dataset import PetDataset
import pytest
from string import printable
from random import choices, randint
from PIL.Image import UnidentifiedImageError


class TestPetDataset:
    def test_one(self, tmp_path, random_string):
        # Test constructor
        empty_path = f"'{str(tmp_path.absolute())}/annotations/list.txt'"
        with pytest.raises(FileNotFoundError) as execinfo:
            PetDataset(tmp_path)
        error_message = f"No such file or directory: {empty_path}"
        assert error_message in str(execinfo.value)

        with pytest.raises(AssertionError):
            PetDataset(tmp_path, standardize=False, classify_by=random_string)

        with pytest.raises(AssertionError):
            PetDataset(tmp_path, standardize=False, image_memory_layout=random_string)  # noqa: E501

    def test_two(self, empty_annotations):
        # Provide dir with empty 'annotations/list.txt'
        dataset = PetDataset(empty_annotations)
        assert ([], []) == dataset.get_data()

        with pytest.raises(StopIteration):
            dataset.__next__()

    def test_three(self, random_annotations):
        # Provide dir with random data in 'annotations/list.txt'
        PetDataset(random_annotations)
        pytest.fail(str(NotImplemented))

    def test_four(self, fake_annotations):
        # Provide dir with valid data in list.txt
        # but nothing valid elsewhere
        dataset = PetDataset(fake_annotations)
        error_message = "No such file or directory: '"
        error_message += str(fake_annotations.absolute())
        with pytest.raises(FileNotFoundError) as execinfo:
            dataset.get_data()
        assert error_message in str(execinfo.value)

    def test_five(self, fake_images):
        # Provide dir with valid data in annotations/list.txt
        # and empty image files in images/
        with pytest.raises(UnidentifiedImageError) as execinfo:
            dataset = PetDataset(fake_images)
            dataset.get_data()
        error_message = "cannot identify image file '"
        error_message += str(fake_images.absolute())
        assert error_message in str(execinfo.value)


@pytest.fixture(scope="function")
def empty_annotations(tmp_path):
    (tmp_path / 'annotations').mkdir()
    (tmp_path / 'annotations' / 'list.txt').touch()
    return tmp_path


@pytest.fixture(scope="function")
def fake_annotations(empty_annotations):
    with open(empty_annotations / 'annotations' / 'list.txt', 'w') as f:
        print("Abyssinian_100 1 1 1", file=f)
    return empty_annotations


@pytest.fixture(scope="function")
def fake_images(fake_annotations):
    (fake_annotations / 'images').mkdir()
    (fake_annotations / 'images' / 'Abyssinian_100.jpg').touch()
    return fake_annotations


@pytest.fixture(scope="function")
def random_annotations(empty_annotations):
    return empty_annotations


@pytest.fixture(scope="function")
def random_string():
    return "".join(choices(printable, k=randint(0, 1000)))
