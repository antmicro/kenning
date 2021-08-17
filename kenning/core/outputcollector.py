"""
Provides an API for processing and returning data from models and dataprovider
"""

from typing import Any
import argparse


class OutputCollector(object):
    def __init__(self):
        pass

    @classmethod
    def form_argparse(cls):
        """
        Creates argparse parser for the OutputCollector object.

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
        group = parser.add_argument_group(title='OutputCollector arguments')

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
        OutputCollector : object of class OutputCollector
        """
        return cls()

    def process_output(self, input_data: Any, output_data: Any):
        """
        Returns the infered data back to the specific place/device/connection

        Eg. it can save a video file with bounding boxes on objects or stream
        it via a TCP connection, or just show it on screen

        Parameters
        ----------
        input_data : Any
            Data collected from Datacollector that was processed by the model
        output_data : Any
            Data returned from the model
        """
        raise NotImplementedError

    def detach_from_output(self):
        """
        Detaches from the output during shutdown
        """
        raise NotImplementedError

    def should_close(self) -> bool:
        """
        Checks if a specific exit condition was reached

        This allows the OutputCollector to close gracefully if an exit
        condition was reached, eg. when a key was pressed.

        Returns
        -------
        bool : True if exit condition was reached to break the loop
        """
        raise NotImplementedError
