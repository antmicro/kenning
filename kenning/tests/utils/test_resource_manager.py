from pathlib import Path

import pytest

from kenning.utils.resource_manager import (
    ResourceManager,
    ResourceURI,
    Resources,
)


MAGIC_WAND_MODEL_URI = ResourceURI(
    'kenning:///models/classification/magic_wand.h5'
)
PERSON_DETECTION_MODEL_URI = ResourceURI(
    'kenning:///models/classification/person_detect.tflite'
)


@pytest.fixture(scope='function', autouse=True)
def clear_cache():
    ResourceManager().max_cache_size = ResourceManager.MAX_CACHE_SIZE
    ResourceManager().clear_cache()


@pytest.fixture
def download_path() -> Path:
    download_path = pytest.test_directory / 'resources' / 'model.h5'
    if download_path.exists():
        download_path.unlink()

    return download_path


@pytest.mark.xdist_group(name='cache_test')
class TestResourceManager:
    def test_download_resource(self, download_path: Path):
        """
        Tests if resource manager properly downloads model file.
        """
        MAGIC_WAND_MODEL_URI.get_resource(download_path)

        assert download_path.is_file()
        assert download_path.with_suffix(
            download_path.suffix + f'.{ResourceManager.HASHING_ALGORITHM}'
        ).is_file()

    def test_use_cached_file_if_available(self, download_path: Path):
        """
        Tests if resource manager uses cached file when its valid.
        """
        MAGIC_WAND_MODEL_URI.get_resource(download_path)

        mtime = download_path.stat().st_mtime

        MAGIC_WAND_MODEL_URI.get_resource(download_path)

        assert mtime == download_path.stat().st_mtime

    def test_download_file_when_checksum_is_invalid(self, download_path: Path):
        """
        Tests if resource manager downloads file again when cached file is
        invalid.
        """
        MAGIC_WAND_MODEL_URI.get_resource(download_path)

        mtime = download_path.stat().st_mtime

        with open(download_path, 'wb') as downloaded_file:
            downloaded_file.write(b'test')

        MAGIC_WAND_MODEL_URI.get_resource(download_path)

        assert mtime != download_path.stat().st_mtime

    def test_list_cached_files_should_return_all_cached_files(self):
        """
        Tests if list_cached_files returns list of all cached files.
        """
        resource_manager = ResourceManager()

        model_1 = MAGIC_WAND_MODEL_URI.get_resource()

        assert model_1 in resource_manager.list_cached_files()

        model_2 = PERSON_DETECTION_MODEL_URI.get_resource()

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

        MAGIC_WAND_MODEL_URI.get_resource()
        PERSON_DETECTION_MODEL_URI.get_resource()

        assert len(resource_manager.list_cached_files()) == 2

        resource_manager.clear_cache()

        assert len(resource_manager.list_cached_files()) == 0

    def test_download_should_fail_when_insufficient_cache_size(self):
        """
        Tests if download fails when cache max size is too small.
        """
        ResourceManager().max_cache_size = 100

        with pytest.raises(ValueError):
            MAGIC_WAND_MODEL_URI.get_resource()

    def test_get_file_resource(self):
        """
        Tests if path to file is properly interpreted.
        """
        resource = ResourceURI(
            str(pytest.test_directory / 'test_file.txt')
        ).get_resource()

        assert resource == pytest.test_directory.resolve() / 'test_file.txt'

        resource = ResourceURI('~/test_file.txt').get_resource()

        assert resource == Path().home() / 'test_file.txt'

    def test_add_custom_schema(self):
        """
        Tests adding custom schema.
        """
        target_path = pytest.test_directory

        ResourceManager().add_custom_url_schemes(
            {'customschema': lambda path: target_path / path[1:]}
        )

        expected_path = target_path / 'test_file.txt'

        assert (
            ResourceURI('customschema:///test_file.txt').get_resource()
            == expected_path
        )

        expected_path = target_path / 'test/test_file.txt'

        assert (
            ResourceURI('customschema:///test/test_file.txt').get_resource()
            == expected_path
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
        resources = Resources(
            {'model': MAGIC_WAND_MODEL_URI}
        )

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

    def test_resource_is_downloaded_when_accessed(self):
        resources = Resources(
            {'model': MAGIC_WAND_MODEL_URI}
        )
        print(ResourceManager().max_cache_size)
        model_path = resources['model']

        assert model_path.is_file()

        model_path.unlink()

        _ = resources['model']

        assert model_path.is_file()
