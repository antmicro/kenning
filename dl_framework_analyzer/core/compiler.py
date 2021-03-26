"""
Provides an API for model compilers.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple


class ModelCompiler(object):
    """
    Compiles the given model to a different format or runtime.
    """

    def __init__(self, compiled_model_path: str):
        """
        Prepares the ModelCompiler object.

        Parameters
        ----------
        compiled_model_path : str
            Path to file where the compiled model should be saved
        """
        self.compiled_model_path = compiled_model_path

    @classmethod
    def form_argparse(cls):
        """
        Creates argparse parser for the ModelCompiler object.

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
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        args : arguments from ArgumentParser object

        Returns
        -------
        ModelCompiler : object of class ModelCompiler
        """
        return cls(args.compiled_model_path)

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
