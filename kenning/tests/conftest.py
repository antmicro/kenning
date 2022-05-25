import pytest
from random import randint
from PIL import Image


@pytest.fixture
def empty_dir(tmp_path):
    (tmp_path / 'annotations').mkdir()
    (tmp_path / 'annotations' / 'list.txt').touch()
    return tmp_path


@pytest.fixture
def fake_images(empty_dir):
    def random_one_hot():
        return f'{randint(0, 1)} {randint(0, 1)} {randint(0, 1)}'
    amount = 100

    with open(empty_dir / 'annotations' / 'list.txt', 'w') as f:
        [print(f'image_{i} {random_one_hot()}', file=f) for i in range(amount)]
    (empty_dir / 'images').mkdir()
    for i in range(amount):
        file = (empty_dir / 'images' / f'image_{i}.jpg')
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        img = Image.new(mode='RGB', size=(5, 5), color=color)
        img.save(file, 'JPEG')
    return empty_dir
