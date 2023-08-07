# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
ROS2-based inference communication protocol.

Requires 'rclpy' and 'kenning_computer_vision_msgs' packages to be sourced
in the environment.
"""

from kenning.core.runtimeprotocol import RuntimeProtocol, Message, \
        MessageType, ServerStatus

from kenning_computer_vision_msgs.srv import RuntimeProtocolSrv

import rclpy
from rclpy.node import Node


class ROS2Protocol(RuntimeProtocol):
    """
    A ROS2-based runtime protocol for communication.
    It supports only a client side of the runtime protocol.

    Protocol is implemented using ROS2 services.
    """

    arguments_structure = {
            'node_name': {
                'description': 'Name of the ROS2 node',
                'type': str,
                'required': True,
            },
            'service_name': {
                'description': 'Name of the service to communicate via',
                'type': str,
                'required': True,
            },
    }

    def __init__(self,
                 node_name: str,
                 service_name: str,
                 ):
        """
        Initialize the ROS2Protocol.

        Parameters
        ----------
        node_name : str
            Name of the ROS2 node.
        service_name : str
            Name of the service for communication using RuntimeProtocolSrv.
        """

        # ROS2 node
        self.node = None
        self.node_name = node_name

        # Last message's future
        self.future = None

        # Communication service
        self.communication_client = None
        self.service_name = service_name

        RuntimeProtocol.__init__(self)

    def log_debug(self, message: str):
        """
        Sends the debug message to loggers.

        Parameters
        ----------
        message : str
            Message to be sent.
        """
        self.log.debug(message)
        if self.node is not None:
            self.node.get_logger().debug(message)

    def log_error(self, message: str):
        """
        Sends the error message to loggers.

        Parameters
        ----------
        message : str
            Message to be sent.
        """
        self.log.error(message)
        if self.node is not None:
            self.node.get_logger().error(message)

    def convert_to_srv(self, message: Message) -> RuntimeProtocolSrv.Request:
        """
        Converts the runtime protocol message to a ROS2 service request.

        Parameters
        ----------
        message : Message
            Runtime protocol message to be converted.

        Returns
        -------
        RuntimeProtocolSrv.Request :
            Converted service request.
        """
        request = RuntimeProtocolSrv.Request()
        request.message_type = message.message_type.value
        request._data = bytearray(message.payload)

        return request

    def convert_from_srv(self, srv: RuntimeProtocolSrv.Response) -> Message:
        """
        Converts the ROS2 service response to a runtime protocol message.

        Parameters
        ----------
        srv : RuntimeProtocolSrv.Response
            Service response to be converted.

        Returns
        -------
        Message :
            Converted message according to the runtime protocol specification.
        """
        runtime_message = Message(MessageType(srv.message_type),
                                  b''.join(srv.data))
        return runtime_message

    def initialize_client(self):
        self.log_debug(f'Initializing service client node {self.node_name}')

        # Create node
        if not rclpy.ok():
            rclpy.init()
        self.node = Node(self.node_name)

        # Create service client
        self.communication_client = self.node.create_client(
                RuntimeProtocolSrv,
                self.service_name
        )

        if not self.communication_client.wait_for_service(timeout_sec=1.0):
            self.log_error('Communication service is not available')
            return False

        # Send initialization request
        if not self.send_message(Message(MessageType.OK)):
            self.log_error('Failed to send initialization request')
            return False

        status, message = self.receive_message()
        if status != ServerStatus.DATA_READY:
            self.log_error('Failed to receive initialization response')
            return False
        elif message.message_type != MessageType.OK:
            self.log_error(f'Initialization failed: {message.message_type}')
            return False

        self.log_debug('Successfully initialized client')
        return True

    def send_message(self, message):
        request = self.convert_to_srv(message)
        if not self.communication_client.wait_for_service(timeout_sec=1.0):
            self.log_error('Communication service is not available')
            return False
        self.future = self.communication_client.call_async(request)
        return True

    def receive_message(self, timeout=None):
        if not self.communication_client.wait_for_service(timeout_sec=1.0):
            self.log_error('Communication service is not available')
            return ServerStatus.DISCONNECTED, None
        elif self.future is None:
            self.log_error('No future to wait for')
            return ServerStatus.ERROR, None

        if timeout is None or timeout > 0:
            rclpy.spin_until_future_complete(self.node, self.future,
                                             timeout_sec=timeout)
        else:
            rclpy.spin_once(self.node, timeout_sec=0)

        # Process the response
        response = self.future.result()
        if not response:
            self.log_debug('No response from communication service')
            return ServerStatus.NOTHING, None

        self.future = None
        message = self.convert_from_srv(response)
        return ServerStatus.DATA_READY, message

    def disconnect(self):
        self.log_debug('Disconnecting node')
        self.node.destroy_node()
        self.log_debug('Successfully disconnected')
        pass
