from kenning.datasets.pet_dataset import PetDataset
import pytest
from string import printable
from random import choices, randint
from PIL import Image


class TestPetDataset:
    def test_one(self, tmp_path):
        # Test constructor
        empty_path = f"'{str(tmp_path.absolute())}/annotations/list.txt'"
        with pytest.raises(FileNotFoundError) as execinfo:
            PetDataset(tmp_path)
        error_message = f"No such file or directory: {empty_path}"
        assert error_message in str(execinfo.value)

        with pytest.raises(AssertionError):
            PetDataset(tmp_path, standardize=False,
                       classify_by=random_string())

        with pytest.raises(AssertionError):
            PetDataset(tmp_path, standardize=False,
                       image_memory_layout=random_string())

    def test_two(self, empty_annotations):
        # Provide dir with empty 'annotations/list.txt'
        dataset = PetDataset(empty_annotations)
        assert ([], []) == dataset.get_data()

        with pytest.raises(StopIteration):
            dataset.__next__()

    def test_three(self, random_annotations):
        # Provide dir with random data in 'annotations/list.txt'
        # TODO: Discuss whether there's a better way to distinguish
        # if data is not valid
        with pytest.raises((IndexError, ValueError)):
            PetDataset(random_annotations)

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
        # Provide valid data
        with pytest.raises(AssertionError):
            dataset = PetDataset(fake_images, standardize=False, batch_size=-1)
            dataset.get_data()
        with pytest.raises(AssertionError):
            dataset = PetDataset(fake_images, standardize=False, batch_size=0)
            dataset.get_data()


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
    file = (fake_annotations / 'images' / 'Abyssinian_100.jpg')
    img = Image.new(mode='RGB', size=(5, 5))
    img.save(file, "JPEG")
    return fake_annotations


@pytest.fixture(scope="function")
def random_annotations(empty_annotations):
    with open(empty_annotations / 'annotations' / 'list.txt', 'w') as f:
        [print(random_string(), file=f) for _ in range(100)]
    return empty_annotations


def random_string():
    return "".join(choices(printable, k=randint(0, 1000)))
