"""
TCP-based inference communication protocol.
"""

import socket
import selectors
import kenning.utils.logger as logger
from typing import Tuple
import json
from typing import Optional, List
import time

from kenning.core.runtimeprotocol import RuntimeProtocol
from kenning.core.runtimeprotocol import MessageType
from kenning.core.runtimeprotocol import ServerStatus
from kenning.core.measurements import Measurements
from kenning.core.measurements import MeasurementsCollector


class NetworkProtocol(RuntimeProtocol):
    """
    A TCP-based runtime protocol.

    Protocol is implemented using BSD sockets and selectors-based pooling.

    Every message in the network protocol has a format:

    <num-bytes><msg-type>[<data>]

    Where:

    * num-bytes - tells the size of <msg-type>[<data>] part of the message,
      in bytes
    * msg-type - the type of the message. For message types check the
      MessageType enum from kenning.core.runtimeprotocol
    * <data> - optional data that comes with the message of MessageType
    """
    def __init__(
            self,
            host: str,
            port: int,
            packet_size: int = 4096,
            endianness: str = 'little'):
        """
        Initializes NetworkProtocol.

        Parameters
        ----------
        host : str
            host for the TCP connection
        port : int
            port for the TCP connection
        packet_size : int
            receive packet sizes
        endiannes : str
            endianness of the communication
        """
        self.host = host
        self.port = port
        self.collecteddata = bytes()
        self.endianness = endianness
        self.log = logger.get_logger()
        self.selector = selectors.DefaultSelector()
        self.serversocket = None
        self.socket = None
        self.packet_size = packet_size
        super().__init__()

    @classmethod
    def form_argparse(cls):
        parser, group = super().form_argparse()
        group.add_argument(
            '--host',
            help='The address to the target device',
            type=str,
            required=True
        )
        group.add_argument(
            '--port',
            help='The port for the target device',
            type=int,
            required=True
        )
        group.add_argument(
            '--packet-size',
            help='The maximum size of the received packets, in bytes.',
            type=int,
            default=4096
        )
        group.add_argument(
            '--endianness',
            help='The endianness of data to transfer',
            choices=['big', 'little'],
            default='little'
        )
        return parser, group

    @classmethod
    def from_argparse(cls, args):
        return cls(
            args.host,
            args.port,
            args.packet_size,
            args.endianness
        )

    def accept_client(self, socket, mask) -> Tuple['ServerStatus', Optional[bytes]]:  # noqa: E501
        """
        Accepts the new client.

        Parameters
        ----------
        socket : new client's socket
        mask : selector mask

        Returns
        -------
        Tuple['ServerStatus', bytes] : client addition status
        """
        sock, addr = socket.accept()
        if self.socket is not None:
            self.log.debug(f'Connection already established, rejecting {addr}')
            sock.close()
            return ServerStatus.CLIENT_IGNORED, None
        else:
            self.socket = sock
            self.log.info(f'Connected client {addr}')
            self.socket.setblocking(False)
            self.selector.register(
                self.socket,
                selectors.EVENT_READ | selectors.EVENT_WRITE,
                self.receive_data
            )
            return ServerStatus.CLIENT_CONNECTED, None

    def initialize_server(self):
        self.serversocket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        self.serversocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.serversocket.setblocking(0)
        self.serversocket.bind((self.host, self.port))
        self.serversocket.listen(1)
        self.selector.register(
            self.serversocket,
            selectors.EVENT_READ,
            self.accept_client
        )
        return True

    def initialize_client(self):
        self.socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM
        )
        self.socket.connect((self.host, self.port))
        self.selector.register(
            self.socket,
            selectors.EVENT_READ | selectors.EVENT_WRITE,
            self.receive_data
        )
        return True

    def collect_messages(self, data: bytes) -> Tuple['ServerStatus', Optional[List[bytes]]]:  # noqa: E501
        """
        Parses received data and returns collected messages.

        It takes everything received from the client at a current model.
        It stores incomplete messages until they are fully collected.
        Once the message or multiple messages are fully collected, the
        method returns the list of the messages in form of bytes arrays.

        Parameters
        ----------
        data : bytes
            The currently received bytes from the client

        Returns
        -------
        Tuple[ServerStatus, Optional[List[bytes]]] :
            The method returns the status of the communication (NOTHING if
            message is incomplete, DATA_READY if there are messages to process)
            and optionally it returns list of bytes arrays containing separate
            messages.
        """
        self.collecteddata += data
        if len(self.collecteddata) < 4:
            return ServerStatus.NOTHING, None
        datatoload = int.from_bytes(
            self.collecteddata[:4],
            byteorder=self.endianness,
            signed=False
        )
        if len(self.collecteddata) - 4 < datatoload:
            return ServerStatus.NOTHING, None
        messages = []
        while len(self.collecteddata) - 4 >= datatoload:
            self.collecteddata = self.collecteddata[4:]
            message = self.collecteddata[:datatoload]
            messages.append(message)
            self.collecteddata = self.collecteddata[datatoload:]
            if len(self.collecteddata) > 4:
                datatoload = int.from_bytes(
                    self.collecteddata[:4],
                    byteorder=self.endianness,
                    signed=False
                )
        return ServerStatus.DATA_READY, messages

    def receive_data(self, socket, mask) -> Tuple['ServerStatus', Optional[List[bytes]]]:  # noqa: E501
        data = self.socket.recv(self.packet_size)
        if not data:
            self.log.info('Client disconnected from the server')
            self.selector.unregister(self.socket)
            self.socket.close()
            self.socket = None
            return ServerStatus.CLIENT_DISCONNECTED, None
        else:
            return self.collect_messages(data)

    def wait_for_activity(self):
        events = self.selector.select(timeout=1)
        results = []
        for key, mask in events:
            if mask & selectors.EVENT_READ:
                callback = key.data
                code, data = callback(key.fileobj, mask)
                results.append((code, data))
        if len(results) == 0:
            return [(ServerStatus.NOTHING, None)]
        return results

    def wait_send(self, data):
        """
        Wrapper for sending method that waits until write buffer is ready for
        new data.
        """
        ret = self.socket.send(data)
        while True:
            events = self.selector.select(timeout=1)
            for key, mask in events:
                if key.fileobj == self.socket and mask & selectors.EVENT_WRITE:
                    return ret
        return ret

    def send_data(self, data):
        length = (len(data)).to_bytes(4, self.endianness, signed=False)
        packet = length + data
        index = 0
        while index < len(packet):
            ret = self.wait_send(packet[index:])
            if ret < 0:
                return False
            index += ret
        return True

    def send_message(self, messagetype: 'MessageType', data=bytes()) -> bool:
        """
        Sends message of a given type to the other side of connection.

        Parameters
        ----------
        messagetype : MessageType
            The type of the message
        data : bytes
            The additional data for a given message type

        Returns
        -------
        bool : True if succeded
        """
        mt = messagetype.to_bytes()
        return self.send_data(mt + data)

    def parse_message(self, message):
        mt = MessageType.from_bytes(message[:2], self.endianness)
        data = message[2:]
        return mt, data

    def receive_confirmation(self) -> Tuple[bool, Optional[bytes]]:
        """
        Waits until the OK message is received.

        Method waits for the OK message from the other side of connection.

        Returns
        -------
        bool : True if OK received, False otherwise
        """
        while True:
            for status, data in self.wait_for_activity():
                if status == ServerStatus.DATA_READY:
                    if len(data) != 1:
                        # this should not happen
                        # TODO handle this scenario
                        self.log.error('There are more messages than expected')
                        return False, None
                    typ, dat = self.parse_message(data[0])
                    if typ == MessageType.ERROR:
                        self.log.error('Error during uploading input')
                        return False, None
                    if typ != MessageType.OK:
                        self.log.error('Unexpected message')
                        return False, None
                    self.log.debug('Upload finished successfully')
                    return True, dat
                elif status == ServerStatus.CLIENT_DISCONNECTED:
                    self.log.error('Client is disconnected')
                    return False, None
                elif status == ServerStatus.DATA_INVALID:
                    self.log.error('Received invalid packet')
                    return False, None
        return False, None

    def upload_input(self, data):
        self.log.debug('Uploading input')
        self.send_message(MessageType.DATA, data)
        return self.receive_confirmation()[0]

    def upload_model(self, path):
        self.log.debug('Uploading model')
        with open(path, 'rb') as modfile:
            data = modfile.read()
            self.send_message(MessageType.MODEL, data)
            return self.receive_confirmation()[0]

    def request_processing(self):
        self.log.debug('Requesting processing')
        self.send_message(MessageType.PROCESS)
        ret = self.receive_confirmation()[0]
        if not ret:
            return False
        start = time.perf_counter()
        ret = self.receive_confirmation()[0]
        if not ret:
            return False
        duration = time.perf_counter() - start
        measurementname = 'protocol_inference_step'
        MeasurementsCollector.measurements += {
            measurementname: [duration],
            f'{measurementname}_timestamp': [time.perf_counter()]
        }
        return True

    def download_output(self):
        self.log.debug('Downloading output')
        self.send_message(MessageType.OUTPUT)
        return self.receive_confirmation()

    def download_statistics(self):
        self.log.debug('Downloading statistics')
        self.send_message(MessageType.STATS)
        status, dat = self.receive_confirmation()
        measurements = Measurements()
        if status and isinstance(dat, bytes) and len(dat) > 0:
            jsonstr = dat.decode('utf8')
            jsondata = json.loads(jsonstr)
            measurements += jsondata
        return measurements

    def request_success(self, data=bytes()):
        self.log.debug('Sending OK')
        return self.send_message(MessageType.OK, data)

    def request_failure(self):
        self.log.debug('Sending ERROR')
        return self.send_message(MessageType.ERROR)

    def disconnect(self):
        if self.serversocket:
            self.serversocket.close()
        if self.socket:
            self.socket.close()
