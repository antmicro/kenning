"""
Provides an API for processing and returning data from models and dataprovider
"""

from typing import Any
import argparse


class Outputcollector(object):
    def __init__(self):
        pass

    @classmethod
    def form_argparse(cls):
        """
        Creates argparse parser for the Outputcollector object.

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
        group = parser.add_argument_group(title='Dataprovider arguments')

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
        Outputcollector : object of class Outputcollector
        """
        return cls()

    def return_output(self, input_data: Any, output_data: Any):
        """
        Returns the infered data back to the specific place/device/connection

        Eg. it can save a video file with bounding boxes on objects or stream
        it via a TCP connection, or just show it on screen
        """
        raise NotImplementedError

    def visualize_data(self, input_data: Any, output_data: Any) -> Any:
        """
        Method used to add visualizations of the models output

        Eg. draw bounding boxes on frames of video or add a list of
        detected objects in the corner of a frame
        """
        return input_data
