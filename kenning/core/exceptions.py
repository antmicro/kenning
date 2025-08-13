# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
All custom exceptions thrown in Kenning.
"""

from typing import Optional


class KenningError(Exception):
    """
    Generic error in Kenning.
    """

    pass


class ProtocolNotStartedError(KenningError):
    """
    Exception raised by the Protocol, when attempting to use a protocol
    object, that is not initialized with 'initialize_client' or
    'initialize_server' methods.
    """

    pass


class RequestFailure(KenningError):
    """
    Exception for failing requests.
    """

    pass


class ArgsManagerConvertError(KenningError):
    """
    JSON arguments given to Kenning could not be parsed or converted to a valid
    format.
    """

    pass


class ClassInfoInvalidArgument(KenningError):
    """
    Could not retrieve information on the given class, because arguments were
    invalid.
    """

    pass


class AmbiguousImport(KenningError):
    """
    Exception raised by ClassLoader if two or more classes with a provided name
    exist.
    """

    pass


class DownloadError(KenningError):
    """
    Resource cold not be downloaded due to a network error or another unknown
    error.
    """

    pass


class ChecksumVerifyError(KenningError):
    """
    Exception raised when downloaded file has invalid checksum.
    """

    pass


class ModelTooLargeError(KenningError):
    """
    Optimization flow could not be executed, because the model is too large and
    will not fit into the memory of the selected device.
    """

    pass


class PipelineRunnerInvalidConfigError(KenningError):
    """
    Optimization flow could not be executed, because invalid configuration was
    provided to the PipelineRunner.
    """

    pass


class WestExecutionError(KenningError):
    """
    West (Zephyr RTOS build tool) failed, when building a Zephyr-based runtime.
    """

    pass


class RenodeSimulationError(KenningError):
    """
    Exception raised when a Renode command fails.
    """

    pass


class UARTNotFoundInDTSError(KenningError):
    """
    Using runtime working on Zephyr RTOS and UART Protocol, but no UART port
    could be found in the Zephyr device tree.
    """

    pass


class VisualEditorGraphParserError(KenningError):
    """
    When using the Visual Scenario Editor, an error occurred either when
    parsing a graph from the editor to a scenario, or when trying to render
    the scenario as a graph.
    """

    pass


class EdgeTPUCompilerError(KenningError):
    """
    edgetpu_compiler failed to compile the model.
    """

    pass


class Ai8xIzerError(KenningError):
    """
    Raised when ai8xizer.py script fails.
    """

    def __init__(self, model_size: Optional[float] = None, *args):
        super().__init__(*args)
        self.model_size = model_size


class MissingUserMessage(KenningError):
    """
    Exception raised if a template configuration lacks a user message
    under `user_message` key.
    """

    pass


class Ai8xUnsupportedDevice(KenningError):
    """
    Raised if platform, unsupported by ai8x-training framework, is used.
    """

    pass


class IOCompatibilityError(KenningError):
    """
    Attempted to use Kenning modules with incompatible input/output.
    """

    pass


class IOSpecNotFound(KenningError):
    """
    Kenning module IO specification not found.
    """

    pass


class IOSpecWrongFormat(KenningError):
    """
    Kenning module IO specification contains unsupported entries.
    """

    pass


class VideoCaptureDeviceException(KenningError):
    """
    Exception to be raised when VideoCaptureDevice malfunctions
    during frame capture.
    """

    def __init__(self, device_id, message="Video device {} read error"):
        super().__init__(message.format(device_id))


class ROS2DataproviderException(KenningError):
    """
    Exception to be raised when ROS2CameraNodeDataProvider misbehaves.
    """

    pass


class MissingConfigForAutoPyTorchModel(KenningError):
    """
    Raised when required configuration to initialize
    AutoPyTorch model was not provided.
    """

    pass


class ModelExtractionError(KenningError):
    """
    Raised when Kenning model was not properly extracted from AutoPyTorch.
    """

    pass


class ModelClassNotValid(KenningError):
    """
    Raised when provided model class cannot be imported.
    """

    pass


class ModelNotPreparedError(KenningError):
    """
    Exception raised when trying to run the model without loading it first.
    """

    def __init__(
        self,
        msg="Make sure to run prepare_model method before running it.",
        *args,
        **kwargs,
    ):
        super().__init__(msg, *args, **kwargs)


class InputNotPreparedError(KenningError):
    """
    Exception raised when trying to run the model without loading the inputs
    first.
    """

    def __init__(
        self,
        msg="Make sure to run load_input method before running the model.",
        *args,
        **kwargs,
    ):
        super().__init__(msg, *args, **kwargs)


class ModelNotLoadedError(KenningError):
    """Exception raised if a model could not be loaded."""

    pass


class ConversionError(KenningError):
    """
    General purpose exception raised when the model conversion process fails.
    """

    pass


class CompilationError(KenningError):
    """
    General purpose exception raised when the compilation process fails.
    """

    pass


class IOSpecificationNotFoundError(KenningError):
    """
    Exception raised when needed input/output specification can not be found.
    """

    pass


class OptimizedModelSizeError(KenningError):
    """
    Exception raised when retrieving size of the optimized model failed.
    """

    pass


class ModelSizeError(KenningError):
    """
    Exception raised when retrieving size of the model failed.
    """

    pass


class VariableBatchSizeNotSupportedError(KenningError):
    """
    Exception raised when trying to create a model which is not fitted to
    handle variable batch sizes yet.
    """

    def __init__(
        self,
        msg="Inference batch size greater than one not supported for this model.",  # noqa: E501
        *args,
        **kwargs,
    ):
        super().__init__(msg, *args, **kwargs)


class TrainingParametersMissingError(KenningError):
    """
    Exception raised when trying train a model without defined training
    parameters.
    """

    def __init__(
        self,
        params,
        msg="Missing train parameters: {}",
        *args,
        **kwargs,
    ):
        super().__init__(msg.format(", ".join(params)), *args, **kwargs)


class CannotDownloadDatasetError(KenningError):
    """
    Exception raised when dataset cannot be downloaded automatically.
    """

    pass


class AutoMLInvalidSchemaError(KenningError):
    """
    Raised when `arguments_structure` contains not enough information
    or when data are invalid.
    """

    ...


class AutoMLInvalidArgumentsError(KenningError):
    """
    Raised when provided arguments (in `use_model`) do not match with
    model wrapper `arguments_structure`.
    """

    ...


class AutoMLModelSizeError(KenningError):
    """
    Raised when model size is too big.
    """

    def __init__(self, model_size: float, *args):
        super().__init__(*args)
        self.model_size = model_size
