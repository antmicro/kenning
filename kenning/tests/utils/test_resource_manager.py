# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict
from urllib.parse import urlparse

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
    ResourceManager().set_cache_dir(
        pytest.test_directory / 'cache_test'
    )
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
            {'customschema': lambda uri: target_path / uri.path[1:]}
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

    def test_resolve_file_uri(self):
        """
        Tests if file URI is properly resolved
        """

        test_file = (pytest.test_directory / 'test.txt').resolve()
        test_file.touch()
        test_file.write_text('test123')

        resource = ResourceURI(f'file://{str(test_file)}')

        assert resource.is_file()
        assert resource.read_text() == 'test123'

    @pytest.mark.parametrize(
        'url_schema,uri,expected_resolved_uri',
        [
            # check inserting netloc and path
            (
                {'test': 'https://{netloc}{path}'},
                'test://test-netloc/path_a',
                'https://test-netloc/path_a'
            ),
            (
                {'test': 'https://{netloc}{path}'},
                'test://netloc-test/path_a/path_b',
                'https://netloc-test/path_a/path_b'
            ),
            (
                {'test': 'https://{netloc}{path}'},
                'test://test-netloc/path_a?a=1&b=2',
                'https://test-netloc/path_a'
            ),
            (
                {'test': 'https://{netloc}{path}'},
                'test://netloc-test/path_a?a=1&b=2',
                'https://netloc-test/path_a'
            ),
            (
                {'test': 'https://{netloc}{path}'},
                'test:///path_a?a=1&b=2',
                'https:///path_a'
            ),
            # check inserting query
            (
                {'test': 'https://{netloc}{path}?{query}'},
                'test://test-netloc/path_a?a=1&b=2',
                'https://test-netloc/path_a?a=1&b=2'
            ),
            # check inserting path part of given index
            (
                {'test': 'https://{netloc}/{path[0]}'},
                'test://test-netloc/path_a/path_b',
                'https://test-netloc/path_a'
            ),
            (
                {'test': 'https://{netloc}/{path[1]}'},
                'test://test-netloc/path_a/path_b',
                'https://test-netloc/path_b'
            ),
            # check inserting param of given name
            (
                {'test': 'https://{netloc};target={params["branch"]}'},
                'test://test-netloc/path_a/path_b;branch=main;test=1',
                'https://test-netloc;target=main'
            ),
            (
                {'test': 'https://{netloc};target={params["branch"]}'},
                'test://test-netloc/path_a/path_b;test=1;branch=main',
                'https://test-netloc;target=main'
            ),
            # check inserting query param of given name
            (
                {'test': 'https://{netloc}?target={query["branch"]}'},
                'test://test-netloc/path_a/path_b?branch=main&test=1',
                'https://test-netloc?target=main'
            ),
            (
                {'test': 'https://{netloc}?target={query["branch"]}'},
                'test://test-netloc/path_a/path_b?test=1&branch=main',
                'https://test-netloc?target=main'
            ),
            # check inserting path slice
            (
                {'test': 'https://{netloc}/{path[1:]}'},
                'test://test-netloc/path_a/path_b/path_c/path_d',
                'https://test-netloc/path_b/path_c/path_d'
            ),
            (
                {'test': 'https://{netloc}/{path[:2]}'},
                'test://test-netloc/path_a/path_b/path_c/path_d',
                'https://test-netloc/path_a/path_b'
            ),
            # check inserting netloc slice
            (
                {'test': 'https://{netloc[1:]}{path}'},
                'test://a.bb.ccc.dd/path_a',
                'https://bb.ccc.dd/path_a'
            ),
            (
                {'test': 'https://{netloc[::-1]}{path}'},
                'test://a.bb.ccc.dd/path_a',
                'https://dd.ccc.bb.a/path_a'
            ),
            # check more complicated format
            (
                {'test': 'http://{netloc[1]}.{netloc[0]}/{path[2:0:-1]}?{query}'},  # noqa: E501
                'test://test.loc/path_a/path_b/path_c?a=1&b=2&c=3',
                'http://loc.test/path_c/path_b?a=1&b=2&c=3'
            ),
        ]
    )
    def test_resolve_uri_should_properly_insert_parameters(
        self,
        url_schema: Dict[str, str],
        uri: str,
        expected_resolved_uri: str
    ):
        """
        Tests if resolve_uri properly parses pattern string.
        """

        resource_manager = ResourceManager()

        resource_manager.add_custom_url_schemes(url_schema)

        assert (
            resource_manager._resolve_uri(urlparse(uri))
            == expected_resolved_uri
        )

    @pytest.mark.parametrize(
        'url_schema,uri,expected_exception',
        [
            # invalid index format
            (
                {'test': 'https://{netloc}{path[abc]}'},
                'test://test-netloc/path_a',
                ValueError
            ),
            # empty index
            (
                {'test': 'https://{netloc}{path[]}'},
                'test://test-netloc/path_a',
                ValueError
            ),
            # invalid param
            (
                {'test': 'https://{netloc}{unknown_param}'},
                'test://test-netloc/path_a',
                ValueError
            ),
            # whitespaces in index
            (
                {'test': 'https://{netloc}/{path[1 :]}'},
                'test://test-netloc/path_a/path_b/path_c',
                ValueError
            ),
            #  index out of range
            (
                {'test': 'https://{netloc}{path[2]}'},
                'test://test-netloc/path_a',
                IndexError
            ),
            (
                {'test': 'https://{netloc[3]}{path}'},
                'test://test-netloc/path_a',
                IndexError
            ),
            # invalid key
            (
                {'test': 'https://{netloc}?target={query["branch"]}'},
                'test://test-netloc/path_a/path_b?abc=main&test=1',
                KeyError
            ),
            (
                {'test': 'https://{netloc}?target={params["branch"]}'},
                'test://test-netloc/path_a/path_b;abc=main;test=1',
                KeyError
            ),
            #  indexing param that does not support it
            (
                {'test': 'https://{netloc}{path}{query[1]}'},
                'test://test-netloc/path_a?a=1&b=2&c=3',
                ValueError
            ),
        ]
    )
    def test_resolve_uri_should_raise_exception_for_invalid_patterns(
        self,
        url_schema: Dict[str, str],
        uri: str,
        expected_exception: Exception
    ):
        """
        Tests if resolve_uri raises exception for invalid patterns.
        """

        resource_manager = ResourceManager()

        resource_manager.add_custom_url_schemes(url_schema)

        with pytest.raises(expected_exception):
            resource_manager._resolve_uri(urlparse(uri))


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
