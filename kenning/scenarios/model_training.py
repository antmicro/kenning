#!/usr/bin/env python

"""
The script for training models given in ModelWrapper object with dataset given
in Dataset object.
"""

import sys
import argparse
from pathlib import Path

from kenning.utils.class_loader import load_class


def main(argv):
    parser = argparse.ArgumentParser(argv[0], add_help=False)
    parser.add_argument(
        'modelwrappercls',
        help='ModelWrapper-based class with inference implementation to import',  # noqa: E501
    )
    parser.add_argument(
        'datasetcls',
        help='Dataset-based class with dataset to import',
    )

    args, _ = parser.parse_known_args(argv[1:])

    modelwrappercls = load_class(args.modelwrappercls)
    datasetcls = load_class(args.datasetcls)

    parser = argparse.ArgumentParser(
        argv[0],
        parents=[
            parser,
            modelwrappercls.form_argparse()[0],
            datasetcls.form_argparse()[0]
        ]
    )

    parser.add_argument(
        '--batch-size',
        help='The batch size for training',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--learning-rate',
        help='The learning rate for training',
        type=float,
        required=True
    )
    parser.add_argument(
        '--num-epochs',
        help='Number of epochs to train for',
        type=int,
        required=True
    )
    parser.add_argument(
        '--logdir',
        help='Path to the training logs directory',
        type=Path,
        required=True
    )
    parser.add_argument(
        '--verbosity',
        help='Verbosity level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )

    args = parser.parse_args(argv[1:])

    dataset = datasetcls.from_argparse(args)
    model = modelwrappercls.from_argparse(dataset, args, from_file=False)

    args.logdir.mkdir(parents=True, exist_ok=True)

    model.train_model(
        args.batch_size,
        args.learning_rate,
        args.num_epochs,
        args.logdir
    )
    model.save_model(model.modelpath)


if __name__ == '__main__':
    main(sys.argv)
