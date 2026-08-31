# Kenning protocols

{{projecturl}} enables workflows spanning multiple machines, usually utilizing client-server architecture, where the server (also referred to as the target platform or the target device) is running tasks (such as inference or model optimization) on the client’s (host’s) requests, which e.g. delivers models to run or input samples from the dataset.

[Protocol](protocol-api) was created as a uniform interface for communication between two Kenning instances, serving as an abstraction for the underlying communication logic and technology stack.

There are multiple `Protocol` implementations - below is outlined the inheritance hierarchy.

  * [BytesBasedProtocol](bytes-based-protocol) *(abstract - requires further implementations)* - implements `kenning.core.protocol.Protocol` abstract methods using a simple protocol with custom data format.
    Cannot be used on its own (requires implementing the underlying logic of requests and transmissions).
    * [KenningProtocol](kenning-protocol) *(abstract - requires further implementations)* - implements communication logic and flow control for requests and transmissions relying on `BytesBasedProtocol`.
      Introduces `send_data` and `receive_data` abstract methods (for sending and receiving raw bytes) that user needs to implement to send prepared packets over selected mean of communication.
      There are:
      * [NetworkProtocol](network-protocol) - Implements methods `send_data` and `receive_data` using [Python sockets](https://docs.python.org/3/library/socket.html) (TCP/IP protocol).
      * [UARTProtocol](uart-protocol) - Implements methods `send_data` and `receive_data` using the [UART standard](https://ieeexplore.ieee.org/document/7586376).
  * [ROS2Protocol](ros2-protocol) - Implementation based on the [ROS2 framework](https://docs.ros.org/en/rolling/index.html).

:::{note}

Things to take into account:

* Some of the implementations are not complete communication solutions and require implementations of their own.
* Not all implementations are compatible with all platforms (for example Kenning Zephyr Runtime can currently only communicate over UART), some implementations do not support all methods (for example `UARTProtocol` only supports client-side API).
:::

(bytes-based-protocol)=
## BytesBasedProtocol

[BytesBasedProtocol](https://github.com/antmicro/kenning/blob/main/kenning/protocols/bytes_based_protocol.py) is an abstract class implementing all `kenning.core.protocol.Protocol` abstract methods and introducing custom protocol message format.
In addition to initialization and disconnecting, `BytesBasedProtocol` requires implementing following methods:

* [transmit](kenning.protocols.bytes_based_protocol.BytesBasedProtocol.transmit)
* [transmit_blocking](kenning.protocols.bytes_based_protocol.BytesBasedProtocol.transmit_blocking)
* [request](kenning.protocols.bytes_based_protocol.BytesBasedProtocol.request)
* [request_blocking](kenning.protocols.bytes_based_protocol.BytesBasedProtocol.request_blocking)
* [listen](kenning.protocols.bytes_based_protocol.BytesBasedProtocol.listen)
* [listen_blocking](kenning.protocols.bytes_based_protocol.BytesBasedProtocol.listen_blocking)
* [event_active](kenning.protocols.bytes_based_protocol.BytesBasedProtocol.event_active)
* [kill_event](kenning.protocols.bytes_based_protocol.BytesBasedProtocol.kill_event)

These methods allow for requesting, sending and receiving payload (`bytes`), along with a set of flags (defined by enum `kenning.protocols.bytes_based_protocol.TransmissionFlag`).

Each request and transmission is identified by a `MessageType`, which serves to differentiate between concurrent communication streams (for example: the client is requesting inference output and at the same time the server is sending logs), as well as to identify the correct server callback.
This class does not work as a standalone communication solution - aforementioned abstract methods, as well as `kenning.core.protocols.Protocol` methods [initialize_client](kenning.core.protocol.Protocol.initialize_client), [initialize_server](kenning.core.protocol.Protocol.initialize_server) and [disconnect](kenning.core.protocol.Protocol.disconnect) have to be provided by its implementations.

### BytesBasedProtocol implementations

* [KenningProtocol](kenning-protocol)
  * [NetworkProtocol](network-protocol)
  * [UARTProtocol](uart-protocol)

### Available MessageType values

* `PING` - Message for checking the connection.
* `STATUS` - Message for reporting server status.
* `DATA` - Message contains inference input.
* `MODEL` - Message contains model to load.
* `PROCESS` - Message means the data should be processed.
* `OUTPUT` - Host requests the output from the target.
* `STATS` - Host requests the inference statistics from the target.
* `IO_SPEC` - Message contains io specification to load.
* `OPTIMIZERS` - Message contains optimizers config.
* `OPTIMIZE_MODEL` - Host requests model optimization and receives optimized model.
* `RUNTIME` - Message contains runtime that should be used for inference (i.e. LLEXT binary).
* `UNOPTIMIZED_MODEL` - Message contains an unoptimized model.
* `LOGS` - Log messages sent from the target device (server).

### Available TransmissionFlags

* `SUCCESS` - Transmission is informing about a success (for example successful inference).
* `FAIL`- Transmission is informing about a failure.
* `IS_HOST_MESSAGE`- Not set if the transmission was sent by the target device. Set otherwise.
* `IS_KENNING`- Messages sent by Kenning.
* `IS_ZEPHYR`- Messages sent by Kenning Zephyr Runtime.

### Protocol specification

Sending requests from the client side:

* [upload_runtime](kenning.core.protocol.Protocol.upload_runtime) - Sends request: `MessageType.RUNTIME`, binary data from the file as payload.
  Expects a transmission with `TransmissionFlag.SUCCESS` and either: `TransmissionFlag.IS_ZEPHYR` or `TransmissionFlag.IS_KENNING` and `ServerAction.UPLOADING_RUNTIME` as payload.
* [upload_io_specification](kenning.core.protocol.Protocol.upload_io_specification) - Sends request, `MessageType.IO_SPEC`, binary data from the file as payload.
  Expects a transmission with `TransmissionFlag.SUCCESS` and either: `TransmissionFlag.IS_ZEPHYR` or `TransmissionFlag.IS_KENNING` and `ServerAction.UPLOADING_IOSPEC` as payload.
* [upload_model](kenning.core.protocol.Protocol.upload_model) - Sends request, `MessageType.MODEL`, binary data from the file as payload.
  Expects a transmission with `TransmissionFlag.SUCCESS` and either: `TransmissionFlag.IS_ZEPHYR` or `TransmissionFlag.IS_KENNING` and `ServerAction.UPLOADING_MODEL` as payload.
* [upload_input](kenning.core.protocol.Protocol.upload_input) - Sends request, `MessageType.DATA`, raw bytes as payload.
  Expects a transmission with `TransmissionFlag.SUCCESS` and either: `TransmissionFlag.IS_ZEPHYR` or `TransmissionFlag.IS_KENNING` and `ServerAction.UPLOADING_INPUT` as payload.
* [request_processing](kenning.core.protocol.Protocol.request_processing) - Sends request, `MessageType.PROCESS`, no payload.
  Expects a transmission with `TransmissionFlag.SUCCESS` and either: `TransmissionFlag.IS_ZEPHYR` or `TransmissionFlag.IS_KENNING` and `ServerAction.PROCESSING_INPUT` as payload.
* [download_output](kenning.core.protocol.Protocol.download_output) - Sends request `MessageType.OUTPUT`, no payload.
  Expects a transmission with `TransmissionFlag.SUCCESS` and non-empty payload.
* [download_statistics](kenning.core.protocol.Protocol.download_statistics) - Sends request `MessageType.STATS`, no payload.
  Expects a transmission with `TransmissionFlag.SUCCESS` and non-empty payload.
* [upload_optimizers](kenning.core.protocol.Protocol.upload_optimizers) - Sends request `MessageType.OPTIMIZERS`, JSON serialized to string and encoded as payload.
  Expects a transmission with `TransmissionFlag.SUCCESS` and either: `TransmissionFlag.IS_ZEPHYR` or `TransmissionFlag.IS_KENNING` and `ServerAction.UPLOADING_OPTIMIZERS` as payload.
* [request_optimization](kenning.core.protocol.Protocol.request_optimization) - Sends request `MessageType.UNOPTIMIZED_MODEL` with binary data from the model file as payload.
After receiving a response client sends a request `MessageType.OPTIMIZE_MODEL` with no payload.
  Expects a transmission with compiled model as payload and `TransmissionFlag.SUCCESS` set.

Serving requests on the server side, by `MessageType` (after the `serve` method has been called to provide callbacks):

* Request with message type: `IO_SPEC`, `MODEL`, `DATA`, `PROCESS`, `OPTIMIZERS` or  `UNOPTIMIZED_MODEL` - Calls appropriate server callback.
  Sends a transmission with the `ServerAction` (from the `ServerStatus` returned by the callback) as payload, as well as the `TransmissionFlag.IS_KENNING` set.
  If the `success` field of the `ServerStatus` is set to `True`, `TransmissionFlag.SUCCESS` is set. Otherwise `TransmissionFlag.FAIL` is set.
* Request with message type: `OPTIMIZE_MODEL`, `OUTPUT`, `STATS` - Calls appropriate callback.
  If the callback returned `ServerStatus` with the `success` field set to true, sends a transmission with the bytes returned by the callback as payload, as well as `TransmissionFlag.IS_KENNING` and `TransmissionFlag.SUCCESS` set.
  Otherwise sends a transmission with the `ServerAction` as payload, as well as `TransmissionFlag.IS_KENNING` and `TransmissionFlag.FAIL` set.

Logs are sent by the server as unprompted transmissions (`MessageType.LOGS`).

Sending multiple log messages per transmission is allowed.

Payload format:

`<message 1 size (1 byte)><message 1 ASCII string><message 2 ASCII size (1 byte)><message 2 ASCII string>...<message n size (1byte)><message n ASCII string>.`


**Methods:**

```{eval-rst}
.. autoclass:: kenning.protocols.bytes_based_protocol.BytesBasedProtocol
   :show-inheritance:
   :members:
```

(kenning-protocol)=
## KenningProtocol

[KenningProtocol](https://github.com/antmicro/kenning/blob/main/kenning/protocols/bytes_based_protocol.py) is an abstract class implementing the abstract methods of [BytesBasedProtocol](bytes-based-protocol) (so methods such as `request`, `transmit_blocking`) using abstract methods [receive_data](kenning.protocols.kenning_protocol.KenningProtocol.receive_data) and [send_data](kenning.protocols.kenning_protocol.KenningProtocol.send_data) (which have to be provided by implementations of this class).

Therefore, it uses a data link that allows for sending/receiving raw bytes and builds a protocol over it that allows for asynchronous, simultaneous communication through multiple channels (each `MessageType` is essentially a separate channel).


### KenningProtocol implementations

* [NetworkProtocol](network-protocol)
* [UARTProtocol](uart-protocol)

### Message structure

| Size   | 6 bits   | 2 bits       | 8 bits   | 16 bits | 0 or 32 bits          | 0-n bits |
| ------ | -------- | ------------ | -------- | ------- | --------------------- | -------- |
| Offset | 0-5      | 6-7          | 8-15     | 16-31   | 32-63                 | 64       |
| Field  | Msg Type | Flow control | Checksum | Flags   | Payload size          | Payload  |

### Flow control values

* `REQUEST` - message is a request for transmission
* `REQUEST_RETRANSMIT`- message is a request to repeat the last message (because of invalid checksum)
* `ACKNOWLEDGE` - message is an acknowledgement of the last message
* `TRANSMISSION` - message is part of a transmission

### Checksum

Bit-wise XOR of all bytes in the message (except for the checksum itself) with `0x4B` byte.

### Flags

| Flag   | Offset in the `Flags` field  | Meaning for `REQUEST` and `TRANSMISSION` messages   | Meaning for `ACKNOWLEDGE` messages   |
| ------ | -------- | ------------ | -------- |
| `SUCCESS` | 0      |    can be set/read as a ```TransmissionFlag```     | last message was received and accepted    |
|  `FAIL`   | 1 |  can be set/read as a ```TransmissionFlag``` | last message was received, but it was rejected (`ACKNOWLEDGE` with a `FAIL` flag is used to deny a request or reject a transmission) |
|  `IS_HOST_MESSAGE`   | 2 |  can be set/read as a ```TransmissionFlag``` | - |
|  `HAS_PAYLOAD`   | 3 | message has payload | - |
|  `FIRST`   | 4 |  first message in this transmission/request | - |
|  `LAST`   | 5| last message in this transmission/request (it is possible for a message to be both `FIRST` and `LAST`, if the transmission/request only has a single message | - |
|  `IS_KENNING`   | 6 |  can be set/read as a ```TransmissionFlag``` | - |
|  `IS_ZEPHYR`   | 7 |  can be set/read as a ```TransmissionFlag``` | - |
|  `SERIALIZED` (only for ```MessageType.IO_SPEC```) | 12 |  can be set/read as a ```TransmissionFlag``` | - |

As you can see, certain flags are used by `KenningProtocol` for flow-control and protocol logic.
Other flags are available to the protocol user (like `BytesBasedProtocol` class), as `TransmissionFlag`.
And with some flags, that depends on what message it is.
Some flags are only present for a specific message type (these are carried in the last 4 bits of the `Flags` field).

### Payload size

Size of the payload in bytes.

### Example communication scenarios

:::{note}
Transmission flags are omitted.
:::

Request (no payload) denied:
* A -> B `MessageType.OUTPUT`, `REQUEST`, (`FIRST`, `LAST`)
* A <- B `MessageType.OUTPUT`, `ACKNOWLEDGE`, (`FAIL`)

Unprompted single-message transmission:
* A -> B `MessageType.LOGS`, `TRANSMISSION`, (`HAS_PAYLOAD`, `FIRST`, `LAST`)

Request (no payload) with a multi-message transmission as a response:
* A -> B `MessageType.STATS`, `REQUEST`, (`FIRST`, `LAST`)
* A <- B `MessageType.STATS`, `TRANSMISSION`, (`HAS_PAYLOAD`, `FIRST`)
* A <- B `MessageType.STATS`, `TRANSMISSION`, (`HAS_PAYLOAD`)
* A <- B `MessageType.STATS`, `TRANSMISSION`, (`HAS_PAYLOAD`)
* A <- B `MessageType.STATS`, `TRANSMISSION`, (`HAS_PAYLOAD`, `LAST`)

Multi-message request with payload with a single-message transmission as a response:
* A -> B `MessageType.IO_SPEC`, `REQUEST`, (`HAS_PAYLOAD`, `FIRST`)
* A -> B `MessageType.IO_SPEC`, `REQUEST`, (`HAS_PAYLOAD`)
* A -> B `MessageType.IO_SPEC`, `REQUEST`, (`HAS_PAYLOAD`, `LAST`)
* A <- B `MessageType.STATUS`, `TRANSMISSION`, (`HAS_PAYLOAD`, `FIRST`, `LAST`)

### Methods

```{eval-rst}
.. autoclass:: kenning.protocols.kenning_protocol.KenningProtocol
   :show-inheritance:
   :members:
```

(network-protocol)=
## NetworkProtocol

[NetworkProtocol](https://github.com/antmicro/kenning/blob/main/kenning/protocols/bytes_based_protocol.py) implements abstract methods from [KenningProtocol](kenning-protocol) ([send_data](kenning.protocols.kenning_protocol.KenningProtocol.send_data) and [receive_data](kenning.protocols.kenning_protocol.KenningProtocol.receive_data)), as well as abstract methods from [Protocol](protocol-api), that neither [KenningProtocol](kenning-protocol) nor [BytesBasedProtocol](bytes-based-protocol) implemented (so [disconnect](kenning.core.protocol.Protocol.disconnect), [initialize_client](kenning.core.protocol.Protocol.initialize_client), and [initialize_server](kenning.core.protocol.Protocol.initialize_server)).

It uses [Python sockets](https://docs.python.org/3/library/socket.html).

After calling `initialize_server` it waits for a client to connect and creates a socket. Only one client at a time can be connected.

When a client connects, the server sends a single byte (`0x00`) to the client, as a confirmation and test of the connection.

The client does not respond.

### Methods

```{eval-rst}
.. autoclass:: kenning.protocols.network.NetworkProtocol
   :show-inheritance:
   :members:
```

(uart-protocol)=
## UARTProtocol

[UARTProtocol](https://github.com/antmicro/kenning/blob/main/kenning/protocols/uart.py) implements abstract methods from [KenningProtocol](kenning-protocol) ([send_data](kenning.protocols.kenning_protocol.KenningProtocol.send_data) and [receive_data](kenning.protocols.kenning_protocol.KenningProtocol.receive_data)), as well as abstract methods from [Protocol](protocol-api), that neither [KenningProtocol](kenning-protocol) nor [BytesBasedProtocol](bytes-based-protocol) implemented (so [disconnect](kenning.core.protocol.Protocol.disconnect) and [initialize_client](kenning.core.protocol.Protocol.initialize_client)).

The [initialize_server](kenning.core.protocol.Protocol.initialize_server) method raises `kenning.core.exceptions.NotSupportedError`, since Kenning Server working on UART is not supported.

When initializing, the client sends a `MessageType.PING` request with `TransmissionFlag.SUCCESS`, as a test of the connection and a signal starting a session.

The server responds with a `MessageType.PING` transmission with `TransmissionFlag.SUCCESS` if it accepts the connection and `TransmissionFlag.FAIL` if it does not accept the connection (for example because a client is already connected).

When disconnecting, the client sends a `MessageType.PING` request with `TransmissionFlag.FAIL`.

The server responds with a `MessageType.PING`, `TransmissionFlag.SUCCESS` transmission.

The `UARTProtocol` class overrides some of the `kenning.core.protocol.Protocol` abstract methods, that are already implemented by `BytesBasedProtocol`, in order to change their behaviour:

* [download_statistics](kenning.protocols.uart.UARTProtocol.download_statistics) - It expects the statistics to arrive in a custom format, instead of a serialized JSON.
Statistics are parsed by the `kenning.protocols.uart._parse_stats` function.

* [upload_io_specification](kenning.protocols.uart.UARTProtocol.upload_io_specification) - Instead of sending the input/output specification as a JSON serialized to string, it sends it serialized to a custom format that enables easy mapping of the binary data to a packed C struct.
The format is defined in the form of a dict (`kenning.interfaces.io_spec_serializer.IOSPEC_STRUCT_FIELDS`).

### Methods

```{eval-rst}
.. autoclass:: kenning.protocols.uart.UARTProtocol
   :show-inheritance:
   :members:
```

(ros2-protocol)=
## ROS2Protocol

[ROS2Protocol](https://github.com/antmicro/kenning/blob/main/kenning/protocols/ros2.py) is an implementation based on the [ROS 2 framework](https://docs.ros.org/en/rolling/index.html).

It does not support server-side API methods ([initialize_server](kenning.core.protocol.Protocol.initialize_server),  [serve](kenning.core.protocol.Protocol.serve)), target-side model optimization ([upload_optimizers](kenning.core.protocol.Protocol.upload_optimizers),  [request_optimization](kenning.core.protocol.Protocol.request_optimization)), dynamic runtime changes ([upload_runtime](kenning.core.protocol.Protocol.upload_runtime)) or sending/receiving logs ([listen_to_server_logs](kenning.core.protocol.Protocol.listen_to_server_logs), [start_sending_logs](kenning.core.protocol.Protocol.start_sending_logs), [stop_sending_logs](kenning.core.protocol.Protocol.stop_sending_logs)).

### Methods

```{eval-rst}
.. autoclass:: kenning.protocols.ros2.ROS2Protocol
   :show-inheritance:
   :members:
```
