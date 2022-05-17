from kenning.datasets.pet_dataset import PetDataset
import pytest


class TestPetDataset:
    def test_two(self):
        # Provide empty path
        with pytest.raises(FileNotFoundError) as execinfo:
            PetDataset("/tmp/existing/")
        assert "No such file or directory: '/tmp/existing/annotations/list.txt'" in str(execinfo.value)
