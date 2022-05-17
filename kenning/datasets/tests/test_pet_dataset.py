from kenning.datasets.pet_dataset import PetDataset
import pytest


class TestPetDataset:
    def test_one(self, tmp_path):
        # Provide empty dir
        empty_path = str(tmp_path.absolute())
        with pytest.raises(FileNotFoundError) as execinfo:
            PetDataset(tmp_path)
        error_message = f"No such file or directory: '{empty_path}/annotations/list.txt'"  # noqa: E501
        assert error_message in str(execinfo.value)

    def test_two(self, tmp_path):
        # Provide dir with empty 'annotations/list.txt'
        q = tmp_path / 'annotations'
        q.mkdir()
        q = q / 'list.txt'
        q.touch()
        dataset = PetDataset(tmp_path)
        assert ([], []) == dataset.get_data()
