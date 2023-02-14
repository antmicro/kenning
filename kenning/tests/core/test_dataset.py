import pytest
from typing import Type, Final
import os
import shutil

from kenning.core.dataset import Dataset, CannotDownloadDatasetError
from kenning.utils.class_loader import get_all_subclasses
from kenning.tests.core.conftest import get_reduced_dataset_path
from kenning.tests.core.conftest import get_dataset_download_path


DATASET_SUBCLASSES: Final = get_all_subclasses(
    'kenning.datasets',
    Dataset,
    raise_exception=True
)


@pytest.fixture(scope='function')
def dataset(request):
    dataset_cls = request.param
    try:
        path = get_reduced_dataset_path(dataset_cls)
        dataset = dataset_cls(path, download_dataset=False)
        assert len(dataset.dataX) > 0
    except CannotDownloadDatasetError:
        pytest.xfail('Cannot download dataset.')
    except Exception as e:
        pytest.fail(f'Exception {e}')
    return dataset


class TestDataset:
    @pytest.mark.parametrize('dataset_cls', [
        pytest.param(dataset_cls, marks=[
            pytest.mark.xdist_group(name=f'TestDataset_{dataset_cls.__name__}')
        ])
        for dataset_cls in DATASET_SUBCLASSES
    ])
    def test_folder_does_not_exist(self, dataset_cls: Type[Dataset]):
        """
        Tests throwing exception when there is no folder with data.
        """
        dataset_download_dir = get_dataset_download_path(dataset_cls)
        if dataset_download_dir.exists():
            shutil.rmtree(str(dataset_download_dir), ignore_errors=True)

        try:
            dataset = dataset_cls(dataset_download_dir, download_dataset=False)
            dataset.prepare()
            if 'Random' not in dataset_cls.__name__:
                pytest.fail('No exception thrown')
        except FileNotFoundError:
            pass
        except Exception as e:
            pytest.fail(f'Exception {e}')

    @pytest.mark.skip(reason='avoiding hitting download rate limit')
    @pytest.mark.parametrize('dataset_cls', [
        pytest.param(dataset_cls, marks=[
            pytest.mark.dependency(
                name=f'test_download[{dataset_cls.__name__}]'
            ),
            pytest.mark.xdist_group(name=f'TestDataset_{dataset_cls.__name__}')
        ])
        for dataset_cls in DATASET_SUBCLASSES
    ])
    def test_download(self, dataset_cls: Type[Dataset]):
        """
        Tests dataset downloading.
        """
        dataset_download_dir = get_dataset_download_path(dataset_cls)
        if os.path.isdir(dataset_download_dir):
            shutil.rmtree(dataset_download_dir, ignore_errors=True)

        try:
            dataset = dataset_cls(dataset_download_dir, download_dataset=True)
            assert len(dataset.dataX) > 0
        except CannotDownloadDatasetError:
            pytest.xfail('Cannot download dataset.')
        except Exception as e:
            pytest.fail(f'Exception {e}')

    @pytest.mark.parametrize('dataset', [
        pytest.param(dataset_cls, marks=[
            pytest.mark.xdist_group(name=f'TestDataset_{dataset_cls.__name__}')
        ])
        for dataset_cls in DATASET_SUBCLASSES
    ], indirect=True)
    def test_iterator(self, dataset: Type[Dataset]):
        """
        Tests dataset iteration.
        """
        for i, (x, y) in enumerate(dataset):
            assert x is not None
            assert y is not None
            if i > 10:
                break

        assert len(dataset) > 0

    @pytest.mark.parametrize('dataset', [
        pytest.param(dataset_cls, marks=[
            pytest.mark.xdist_group(name=f'TestDataset_{dataset_cls.__name__}')
        ])
        for dataset_cls in DATASET_SUBCLASSES
    ], indirect=True)
    def test_data_equal_length(self, dataset: Type[Dataset]):
        """
        Tests dataset iteration.
        """
        assert len(dataset) == len(dataset.dataX)
        assert len(dataset.dataX) == len(dataset.dataY)

    @pytest.mark.parametrize('dataset', [
        pytest.param(dataset_cls, marks=[
            pytest.mark.xdist_group(name=f'TestDataset_{dataset_cls.__name__}')
        ])
        for dataset_cls in DATASET_SUBCLASSES
    ], indirect=True)
    def test_train_test_split(self, dataset: Type[Dataset]):
        """
        Tests the `train_test_split_representations` method.
        """
        test_fraction = 0.25
        dataXtrain, dataXtest, dataYtrain, dataYtest = \
            dataset.train_test_split_representations(test_fraction)

        assert len(dataXtrain) > 0
        assert len(dataXtest) > 0
        assert len(dataYtrain) > 0
        assert len(dataYtest) > 0
        assert len(dataXtrain) + len(dataXtest) == len(dataset.dataX)
        assert len(dataYtrain) + len(dataYtest) == len(dataset.dataY)

        tolerance = 1./len(dataset)

        assert (len(dataXtrain)/len(dataset.dataX)
                == pytest.approx(1 - test_fraction, abs=tolerance))
        assert (len(dataYtrain)/len(dataset.dataY)
                == pytest.approx(1 - test_fraction, abs=tolerance))
        assert (len(dataXtest)/len(dataset.dataX)
                == pytest.approx(test_fraction, abs=tolerance))
        assert (len(dataYtest)/len(dataset.dataY)
                == pytest.approx(test_fraction, abs=tolerance))
