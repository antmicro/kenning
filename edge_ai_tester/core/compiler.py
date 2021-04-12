"""
Provides an API for model compilers.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

from edge_ai_tester.core.dataset import Dataset


class ModelCompiler(object):
    """
    Compiles the given model to a different format or runtime.
    """

    def __init__(
            self,
            dataset: Dataset,
            compiled_model_path: str,
            dataset_percentage: float = 1.0):
        """
        Prepares the ModelCompiler object.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to train the model - may be used for quantization
            during compilation stage
        compiled_model_path : str
            Path to file where the compiled model should be saved
        dataset_percentage : float
            If the dataset is used for optimization (quantization), the
            dataset_percentage determines how much of data samples is going
            to be used
        """
        self.dataset = dataset
        self.compiled_model_path = compiled_model_path
        self.dataset_percentage = dataset_percentage

    @classmethod
    def form_argparse(cls, quantizes_model: bool = False):
        """
        Creates argparse parser for the ModelCompiler object.

        Parameters
        ----------
        quantizes_model : bool
            Tells if the compiler quantizes model - if so, flags for
            calibration dataset are enabled

        Returns
        -------
        (ArgumentParser, ArgumentGroup) :
            tuple with the argument parser object that can act as parent for
            program's argument parser, and the corresponding arguments' group
            pointer
        """
        parser = argparse.ArgumentParser(add_help=False)
        group = parser.add_argument_group(title='Compiler arguments')
        group.add_argument(
            '--compiled-model-path',
            help='The path to the compiled model output',
            type=Path,
            required=True
        )
        if quantizes_model:
            group.add_argument(
                '--dataset-percentage',
                help='Tells how much data from dataset (from 0.0 to 1.0) ' +
                     'will be used for calibration dataset',
                type=float,
                default=0.25
            )
        return parser, group

    @classmethod
    def from_argparse(cls, dataset: Dataset, args):
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        dataset : Dataset
            The dataset object that is optionally used for optimization
        args : arguments from ArgumentParser object

        Returns
        -------
        ModelCompiler : object of class ModelCompiler
        """
        if hasattr(args, 'dataset_percentage'):
            return cls(
                dataset,
                args.compiled_model_path,
                args.dataset_percentage
            )
        else:
            return cls(
                dataset,
                args.compiled_model_path
            )

    def compile(
            self,
            inputmodelpath: Path,
            inputshapes: Dict[str, Tuple[int, ...]],
            dtype: str = 'float32'):
        """
        Compiles the given model to a target format.

        The function compiles the model and saves it to the output file.

        The model can be compiled to a binary, a different framework or a
        different programming language.

        The additional compilation parameters that are not derivable from
        the input and output format should be passed in the constructor or via
        argument parsing.

        The compiled model is saved to compiled_model_path

        Parameters
        ----------
        inputmodelpath : Path
            Path to the input model
        inputshapes : Dict[str, Tuple[int, ...]]
            The dictionary with mapping (input name) -> (input shape)
        dtype : str
            The type of input tensors
        """
        raise NotImplementedError
