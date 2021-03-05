"""
Provides an API for model compilers.
"""

import argparse
from typing import Any
from pathlib import Path


class ModelCompiler(object):
    """
    Compiles the given model to a different format or runtime.
    """

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
        return cls()

    def compile(self, inputmodel: Any, outfile: Path):
        """
        Compiles the given model to a target format.

        The function compiles the model and saves it to the output file.

        The model can be compiled to a binary, a different framework or a
        different programming language.

        The additional compilation parameters that are not derivable from
        the input and output format should be passed in the constructor or via
        argument parsing.

        Parameters
        ----------
        inputmodel : Any
            The input model object to be compiled
        outfile : Path
            The path to the output file with compiled model
        """
        raise NotImplementedError
