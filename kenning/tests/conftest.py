import pytest
from random import randint, random, seed
from PIL import Image
seed(12345)


def write_to_dirs(dir_path, amount):
    """
    Creates files under provided 'dir_path' such as 'list.txt' for PetDataset,
    'annotations.csv' and 'classnames.csv' for OpenImagesDataset.
    """
    def three_random_one_hot(i):
        return f'{i%37+1} {randint(0, 1)} {randint(0, 1)}'

    def four_random():
        return f'{random()},{random()},{random()},{random()}'

    with open(dir_path / 'annotations' / 'list.txt', 'w') as f:
        [print(f'image_{i} {three_random_one_hot(i)}', file=f)
         for i in range(amount)]
    with open(dir_path / 'classnames.csv', 'w') as f:
        print('/m/o0fd,person', file=f)
    with open(dir_path / 'annotations.csv', 'w') as f:
        title = 'ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,'
        title += 'IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside'
        print(title, file=f)
        [print(f'image_{i},xclick,/m/o0fd,1,{four_random()},0,0,0,0,0', file=f)
         for i in range(amount)]
    return


@pytest.fixture
def empty_dir(tmp_path):
    (tmp_path / 'annotations').mkdir()
    (tmp_path / 'annotations' / 'list.txt').touch()
    return tmp_path


@pytest.fixture
def fake_images(empty_dir):
    """
    Creates a temporary dir with images.

    Images are located under 'image/' folder.
    """
    amount = 148
    write_to_dirs(empty_dir, amount)
    (empty_dir / 'images').mkdir()
    (empty_dir / 'img').symlink_to(empty_dir / 'images')
    for i in range(amount):
        file = (empty_dir / 'images' / f'image_{i}.jpg')
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        img = Image.new(mode='RGB', size=(5, 5), color=color)
        img.save(file, 'JPEG')

    return (empty_dir, amount)
