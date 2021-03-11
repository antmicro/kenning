from typing import Any


class RuntimeProtocol(object):
    """
    The interface for the communication protocol with the target devices.

    The target device acts as a server in the communication.

    The machine that runs the benchmark and collects the results is the client
    for the target device.

    The inheriting classes for this class implement at least the client-side
    of the communication with the target device.
    """

    @classmethod
    def form_argparse(cls):
        """
        Creates argparse parser for the RuntimeProtocol object.

        Returns
        -------
        (ArgumentParser, ArgumentGroup) :
            tuple with the argument parser object that can act as parent for
            program's argument parser, and the corresponding arguments' group
            pointer
        """
        parser = argparse.ArgumentParser(add_help=False)
        group = parser.add_argument_group(title='Runtime protocol arguments')
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        """
        Constructor wrapper that takes the parameters from argparse args.

        Parameters
        ----------
        args : arguments from RuntimeProtocol object

        Returns
        -------
        RuntimeProtocol : object of class ModelCompiler
        """
        return cls()

    def connect(self) -> bool:
        """
        Connects to the target device.

        Returns
        -------
        bool : True if connected successfully, False otherwise
        """
        raise NotImplementedError

    def serve(self) -> bool:
        """
        Serves the incoming connections in the target device.

        Returns
        -------
        bool : True if the service ended without any errors, False otherwise
        """
        raise NotImplementedError

    def send_data(self, data: Any) -> int:
        """
        Sends data to the target device.

        Parameters
        ----------
        data: Any
            Data to be sent to the target device.

        Returns
        -------
        int : number of successfully sent bytes, -1 if error occured
        """
        raise NotImplementedError

    def receive_data(self) -> int, Any:
        """
        Receives data from the target device.

        Returns
        -------
        int, Any: tuple with the number of received bytes and 
        """
        raise NotImplementedError

    def disconnect(self) -> bool:
        """
        Disconnects from the target device.

        Returns
        -------
        bool : True if disconnected successfully, False otherwise
        """
        raise NotImplementedError
