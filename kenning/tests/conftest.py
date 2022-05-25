import pytest
from random import randint, random
from PIL import Image


@pytest.fixture
def empty_dir(tmp_path):
    (tmp_path / 'annotations').mkdir()
    (tmp_path / 'annotations' / 'list.txt').touch()
    return tmp_path


@pytest.fixture
def fake_images(empty_dir):
    def three_random_one_hot():
        return f'{randint(0, 1)} {randint(0, 1)} {randint(0, 1)}'

    def four_random_range():
        # return f'{random()},{random()},{random()},{random()}'
        return '0.3,0.9,0.2,0.7'
    amount = 100

    with open(empty_dir / 'annotations' / 'list.txt', 'w') as f:
        [print(f'image_{i} {three_random_one_hot()}', file=f)
         for i in range(amount)]
    with open(empty_dir / 'classnames.csv', 'w') as f:
        print('/m/o0fd,person', file=f)
    with open(empty_dir / 'annotations.csv', 'w') as f:
        print('ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside', file=f)  # noqa: E501
        [print(f'image_{i},xclick,/m/o0fd,1,{four_random_range()},0,0,0,0,0', file=f)
         for i in range(amount)]
    (empty_dir / 'images').mkdir()
    for i in range(amount):
        file = (empty_dir / 'images' / f'image_{i}.jpg')
        color = (randint(0, 255), randint(0, 255), randint(0, 255))
        img = Image.new(mode='RGB', size=(5, 5), color=color)
        img.save(file, 'JPEG')
    return empty_dir
