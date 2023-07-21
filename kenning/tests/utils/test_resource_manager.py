from pathlib import Path

import pytest

from kenning.modelwrappers.classification.tflite_magic_wand import (
    MagicWandModelWrapper,
)
from kenning.modelwrappers.classification.tflite_person_detection import (
    PersonDetectionModelWrapper,
)
from kenning.utils.resource_manager import (
    ResourceManager,
    ResourceURI,
    Resources,
)


MAGIC_WAND_MODEL_URI = MagicWandModelWrapper.pretrained_model_uri
PERSON_DETECTION_MODEL_URI = PersonDetectionModelWrapper.pretrained_model_uri


@pytest.fixture(scope='function', autouse=True)
def clear_cache():
    ResourceManager().max_cache_size = ResourceManager.MAX_CACHE_SIZE
    ResourceManager().clear_cache()


@pytest.mark.xdist_group(name='cache_test')
class TestResourceManager:
    def test_download_resource(self):
        """
        Tests if resource manager properly downloads model file.
        """
        model = ResourceURI(MAGIC_WAND_MODEL_URI)

        assert model.is_file()
        assert model.with_suffix(
            model.suffix + f'.{ResourceManager.HASHING_ALGORITHM}'
        ).is_file()

    def test_use_cached_file_if_available(self):
        """
        Tests if resource manager uses cached file when its valid.
        """
        model_1 = ResourceURI(MAGIC_WAND_MODEL_URI)

        model_1_mtime = model_1.stat().st_mtime

        model_2 = ResourceURI(MAGIC_WAND_MODEL_URI)

        assert model_1_mtime == model_2.stat().st_mtime

    def test_download_file_when_checksum_is_invalid(self):
        """
        Tests if resource manager downloads file again when cached file is
        invalid.
        """
        model_1 = ResourceURI(MAGIC_WAND_MODEL_URI)

        model_1_mtime = model_1.stat().st_mtime

        with open(model_1, 'wb') as downloaded_file:
            downloaded_file.write(b'test')

        model_2 = ResourceURI(MAGIC_WAND_MODEL_URI)

        assert model_1_mtime != model_2.stat().st_mtime

    def test_list_cached_files_should_return_all_cached_files(self):
        """
        Tests if list_cached_files returns list of all cached files.
        """
        resource_manager = ResourceManager()

        model_1 = ResourceURI(MAGIC_WAND_MODEL_URI)

        assert model_1 in resource_manager.list_cached_files()

        model_2 = ResourceURI(PERSON_DETECTION_MODEL_URI)

        assert model_2 in resource_manager.list_cached_files()
        assert 2 == len(resource_manager.list_cached_files())

        model_1.unlink()
        assert model_2 in resource_manager.list_cached_files()
        assert 1 == len(resource_manager.list_cached_files())

        model_2.unlink()
        assert 0 == len(resource_manager.list_cached_files())

    def test_clear_cache_should_remove_all_cache(self):
        """
        Tests if clear_cache removes all cached files.
        """
        resource_manager = ResourceManager()

        assert len(resource_manager.list_cached_files()) == 0

        _ = ResourceURI(MAGIC_WAND_MODEL_URI)
        _ = ResourceURI(PERSON_DETECTION_MODEL_URI)

        assert len(resource_manager.list_cached_files()) == 2

        resource_manager.clear_cache()

        assert len(resource_manager.list_cached_files()) == 0

    def test_download_should_fail_when_insufficient_cache_size(self):
        """
        Tests if download fails when cache max size is too small.
        """
        ResourceManager().max_cache_size = 100

        with pytest.raises(ValueError):
            _ = ResourceURI(MAGIC_WAND_MODEL_URI)

    def test_get_file_resource(self):
        """
        Tests if path to file is properly interpreted.
        """
        resource = ResourceURI(str(pytest.test_directory / 'test_file.txt'))

        assert (
            Path(resource) == pytest.test_directory.resolve() / 'test_file.txt'
        )

        resource = ResourceURI('~/test_file.txt')

        assert resource == Path().home() / 'test_file.txt'

    def test_add_custom_schema(self):
        """
        Tests adding custom schema.
        """
        target_path: Path = pytest.test_directory

        ResourceManager().add_custom_url_schemes(
            {'customschema': lambda path: target_path / path[1:]}
        )

        expected_path = target_path / 'test_file.txt'
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text('test123')

        assert (
            ResourceURI('customschema:///test_file.txt').read_text()
            == 'test123'
        )

        expected_path = target_path / 'test/test_file.txt'
        expected_path.parent.mkdir(parents=True, exist_ok=True)
        expected_path.write_text('test12345')

        assert (
            ResourceURI('customschema:///test/test_file.txt').read_text()
            == 'test12345'
        )


@pytest.mark.xdist_group(name='cache_test')
class TestResources:
    def test_initializer_with_dict(self):
        resources = Resources(
            {
                'model_1': MAGIC_WAND_MODEL_URI,
                'model_2': PERSON_DETECTION_MODEL_URI,
            }
        )

        assert 'model_1' in resources
        assert 'model_2' in resources
        assert ('model_2',) in resources
        assert len(resources) == 2

    def test_initializer_with_nested_dict(self):
        resources = Resources(
            {
                'model_1': MAGIC_WAND_MODEL_URI,
                'nested': {
                    'model_2': PERSON_DETECTION_MODEL_URI,
                },
            }
        )

        assert 'model_1' in resources
        assert ('nested', 'model_2') in resources
        assert len(resources) == 2

    def test_invalid_key_raises_exception(self):
        resources = Resources({'model': MAGIC_WAND_MODEL_URI})

        with pytest.raises(KeyError):
            _ = resources['model_1']

        with pytest.raises(KeyError):
            _ = resources['nested', 'model']

        with pytest.raises(KeyError):
            _ = resources['model', 'model']

        with pytest.raises(KeyError):
            _ = resources[tuple()]

        resources = Resources(
            {
                'model_1': MAGIC_WAND_MODEL_URI,
                'nested': {
                    'model_2': PERSON_DETECTION_MODEL_URI,
                },
            }
        )

        with pytest.raises(KeyError):
            _ = resources['model_2']

        with pytest.raises(KeyError):
            _ = resources['nested', 'model_1']
