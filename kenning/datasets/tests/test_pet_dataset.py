from kenning.datasets.pet_dataset import PetDataset
import pytest
from string import printable
from random import choices, randint


class TestPetDataset:
    def test_one(self, tmp_path):
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

    def test_three(self):
        # Provide dir with random data in 'annotations/list.txt'
        pytest.fail(str(NotImplemented))

    def test_four(self, empty_annotations):
        # Provide dir with simple valid data
        PetDataset(empty_annotations)
        pytest.fail(str(NotImplemented))


@pytest.fixture(scope="function")
def empty_annotations(tmp_path):
    (tmp_path / 'annotations').mkdir()
    (tmp_path / 'annotations' / 'list.txt').touch()
    return tmp_path


@pytest.fixture(scope="function")
def random_string():
    return "".join(choices(printable, k=randint(0, 1000)))
