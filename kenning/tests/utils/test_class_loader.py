from kenning.utils.class_loader import get_command
from kenning.utils.class_loader import get_kenning_submodule_from_path
from kenning.utils.class_loader import load_class
import pytest
import kenning.datasets.pet_dataset


@pytest.mark.fast
class TestLoadClass:
    def test_empty_string(self):
        error_message = "not enough values to unpack (expected 2, got 1)"
        with pytest.raises(ValueError) as execinfo:
            load_class("")
        assert error_message in str(execinfo.value)

    def test_invalid_import_path(self):
        error_message = "No module named 'something"
        with pytest.raises(ModuleNotFoundError) as execinfo:
            load_class("something.that.is.not.implemented.here.YET")
        assert error_message in str(execinfo.value)

    def test_valid_import_path(self):
        loaded = load_class("kenning.datasets.pet_dataset.PetDataset")
        assert loaded is kenning.datasets.pet_dataset.PetDataset

    def test_capitalization(self):
        error_message = "No module named"
        with pytest.raises(ModuleNotFoundError) as execinfo:
            load_class("kenning.datasets.PET_DATASET.PetDataset")
        assert error_message in str(execinfo.value)

    def test_multiple_dots(self):
        error_message = "No module named 'kenning.datasets.pet_dataset.'"
        with pytest.raises(ModuleNotFoundError) as execinfo:
            load_class("kenning.datasets.pet_dataset.....PetDataset")
        assert error_message in str(execinfo.value)


@pytest.mark.fast
class TestGetKenningSubmoduleFromPath:
    def test_multiple_kenning(self):
        x = "/tmp/smth/anything/kenning/kenning/kenning/core/"
        y = "kenning.core"
        assert get_kenning_submodule_from_path(x) == y

    def test_not_valid(self):
        with pytest.raises(ValueError) as execinfo:
            x = "/tmp/smth/anything/"
            get_kenning_submodule_from_path(x)
        assert "not in tuple" in str(execinfo.value)

    def test_valid_class_loader_path(self):
        x = "/usr/lib/python3.10/site-packages/kenning/utils/class_loader.py"
        y = "kenning.utils.class_loader"
        assert get_kenning_submodule_from_path(x) == y

    def test_module_name(self):
        x = "/usr/lib/python3.10/site-packages/kenning/"
        y = "kenning"
        assert get_kenning_submodule_from_path(x) == y

    def test_invalid_capitalization(self):
        with pytest.raises(ValueError) as execinfo:
            x = "/usr/lib/python3.10/site-packages/Kenning/utils/class_loader.py"   # noqa: E501
            get_kenning_submodule_from_path(x)
        assert "not in tuple" in str(execinfo.value)


@pytest.mark.fast
class TestGetCommand:
    def test_invalid_capitalization(self):
        with pytest.raises(ValueError) as execinfo:
            x = ['/usr/lib/python3.10/Kenning/utils/class_loader.py', ]
            get_command(x)
        assert "not in tuple" in str(execinfo.value)

    def test_different_flags(self):
        x = ['/usr/lib/python3.10/kenning/scenarios/model_training.py',
             '--help',
             '-h',
             ]
        y = ['python -m kenning.scenarios.model_training \\',
             '    --help \\',
             '    -h'
             ]
        assert get_command(x) == y

    def test_only_file(self):
        x = ['/usr/lib/python3.10/kenning/scenarios/model_training.py', ]
        y = ['python -m kenning.scenarios.model_training', ]
        assert get_command(x) == y

    def test_multiple_flags(self):
        x = ['/usr/lib/python3.10/kenning/scenarios/model_training.py',
             'kenning.modelwrappers.classification.\
tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2',
             'kenning.datasets.pet_dataset.PetDataset',
             '--logdir',
             'build/logs',
             '--dataset-root',
             'build/pet-dataset',
             '--model-path',
             'build/trained-model.h5',
             '--batch-size',
             '32',
             '--learning-rate',
             '0.0001',
             '--num-epochs',
             '50',
             ]
        y = ['python -m kenning.scenarios.model_training \\',
             '    kenning.modelwrappers.classification.\
tensorflow_pet_dataset.TensorFlowPetDatasetMobileNetV2 \\',
             '    kenning.datasets.pet_dataset.PetDataset \\',
             '    --logdir \\',
             '        build/logs \\',
             '    --dataset-root \\',
             '        build/pet-dataset \\',
             '    --model-path \\',
             '        build/trained-model.h5 \\',
             '    --batch-size \\',
             '        32 \\',
             '    --learning-rate \\',
             '        0.0001 \\',
             '    --num-epochs \\',
             '        50',
             ]
        assert get_command(x) == y

    def test_empty_command(self):
        with pytest.raises(IndexError) as execinfo:
            x = []
            get_command(x)
        assert "list index out of range" in str(execinfo.value)

    def test_empty_flags(self):
        x = ['/usr/lib/python3.10/kenning/scenarios/model_training.py',
             '',
             ]
        y = ['python -m kenning.scenarios.model_training']
        assert get_command(x) == y
