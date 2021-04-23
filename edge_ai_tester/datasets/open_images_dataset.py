"""
Open Images Dataset V6 wrapper.

The downloader part of the script is based on Open Images Dataset V6::

    https://raw.githubusercontent.com/openimages/dataset/master/downloader.py
"""

from concurrent import futures
import botocore
import tqdm
import boto3

from edge_ai_tester.core.dataset import Dataset
from edge_ai_tester.utils.logger import download_url

BUCKET_NAME = 'open-images-dataset'
REGEX = r'(test|train|validation|challenge2018)/([a-fA-F0-9]*)'


def check_and_homogenize_one_image(image):
    split, image_id = re.match(REGEX, image).groups()
    yield split, image_id


def check_and_homogenize_image_list(image_list):
    for line_number, image in enumerate(image_list):
        try:
            yield from check_and_homogenize_one_image(image)
        except (ValueError, AttributeError):
            raise ValueError(
                f'ERROR in line {line_number} of the image list.' +
                f'The following image string is not recognized: "{image}".'
            )


def read_image_list_file(image_list_file):
    with open(image_list_file, 'r') as f:
        for line in f:
            yield line.strip().replace('.jpg', '')


def download_one_image(bucket, split, image_id, download_folder):
    try:
        bucket.download_file(
            f'{split}/{image_id}.jpg',
            os.path.join(download_folder, f'{image_id}.jpg')
        )
    except botocore.exceptions.ClientError as exception:
        sys.exit(
            f'ERROR when downloading image `{split}/{image_id}`: ' + 
            f'{str(exception)}'
        )


def download_all_images(download_folder, image_list, num_processes):
    """Downloads all images specified in the input file."""
    bucket = boto3.resource(
        's3', config=botocore.config.Config(
            signature_version=botocore.UNSIGNED
        )
    ).Bucket(BUCKET_NAME)

    if not os.path.exists(download_folder):
        os.makedirs(download_folder)

    try:
        image_list = list(
            check_and_homogenize_image_list(
                read_image_list_file(image_list)
            )
        )
    except ValueError as exception:
        sys.exit(exception)

    progress_bar = tqdm.tqdm(
        total=len(image_list),
        desc='Downloading images',
        leave=True
    )
    with futures.ThreadPoolExecutor(
            max_workers=num_processes) as executor:
        all_futures = [
            executor.submit(
                download_one_image,
                bucket,
                split,
                image_id,
                download_folder
            ) for (split, image_id) in image_list
        ]
        for future in futures.as_completed(all_futures):
            future.result()
            progress_bar.update(1)
    progress_bar.close()

class OpenImagesDatasetV6(Dataset):
    """
    The Open Images Dataset V6

    https://storage.googleapis.com/openimages/web/index.html

    It is a dataset of ~9M images annotated with:

    * image-level labels,
    * object bounding boxes,
    * object segmentation masks,
    * visual relationships,
    * localized narratives.

    *License*: Creative Commons Attribution 4.0 International License.

    *Page*: `Open Images Dataset V6 site <https://storage.googleapis.com/openimages/web/index.html>`_.
    """
    def __init__(
            self,
            root: Path,
            batch_size: int = 1,
            download_dataset: bool = False,
            task: str = 'object_detection',
            classes: str = 'coco',
            download_samples_per_class: int = 200,
            download_annotations_type: str = 'validation'):
        self.task = task
        self.classes = classes
        super().__init__(root, batch_size, download_dataset)

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--task',
            help='The task type',  # noqa: E501
            choices=['object_detection'],
            default='object_detection'
        )
        group.add_argument(
            '--classes',
            help='The path to file with classes to use or class set name',
            choices=['coco'],
            default='coco'
        )
        group.add_argument(
            '--download-samples-per-class',
            help='Number of images per object class',
            type=int,
            default=200
        )
        group.add_argument(
            '--download-annotations-type',
            help='Type of annotations to extract the images from',
            choices=['train', 'validation', 'test'],
            default='validation'
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.dataset_root,
            args.inference_batch_size,
            args.download_dataset,
            args.task,
            args.classes,
            args.download_samples_per_class,
            args.download_annotations_type
        )

    def download_dataset(self):
        self.root.mkdir(parents=True, exist_ok=True)

        classnamesurl = 'https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv'  # noqa: E501
        classnamespath = self.dataset_root / 'classnames.csv'
        
        annotationsurls = {
            'train': 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv',  # noqa: E501
            'validation': 'https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv',  # noqa: E501
            'test': 'https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv'  # noqa: E501
        }
        annotationspath = self.dataset_root / 'annotations.csv'

        download_url(classnamesurl, classnamespath)
        download_url(
            annotationsurls[self.download_annotations_type],
            annotationspath
        )
        
