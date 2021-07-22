"""
Provides an API for gathering and preparing data from external sources
"""

from typing import Any
import argparse


class DataProvider(object):
    def __init__(self):
        """
        Initializes DataProvider object.

        Parameters
        ----------

        """
        self.device = None
        self.prepare()

    @classmethod
    def form_argparse(cls):
        """
        Creates argparse parser for the DataProvider object.

        This method is used to create a list of arguments for the object so
        it is possible to configure the object from the level of command
        line.

        Returns
        -------
        (ArgumentParser, ArgumentGroup) :
            tuple with the argument parser object that can act as parent for
            program's argument parser, and the corresponding arguments' group
            pointer
        """
        parser = argparse.ArgumentParser(add_help=False)
        group = parser.add_argument_group(title='DataProvider arguments')

        return parser, group

    @classmethod
    def from_argparse(cls, args):
        """
        Constructor wrapper that takes the parameters from argparse args.

        This method takes the arguments created in form_argparse and uses them
        to create the object.

        Parameters
        ----------
        args : Dict
            arguments from ArgumentParser object

        Returns
        -------
        DataProvider : object of class DataProvider
        """
        return cls()

    def prepare(self):
        """
        Prepares the source for data gathering depending on the
        source type.

        This will for example initialize the camera and
        set the self.device to it
        """
        raise NotImplementedError

    def get_input(self):
        """
        Gets the sample from device
        """
        raise NotImplementedError

    def preprocess_input(self, data: Any) -> Any:
        """
        Processes the input for inference

        Parameters
        ----------
        data : Any
            the data to be preprocessed

        Returns
        -------
        Any : preprocessed data
        """
        return self.data

    def detach_from_source(self):
        """
        Detaches from the source during shutdown
        """
        raise NotImplementedError
