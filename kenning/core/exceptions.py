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


"""
Miscellaneous errors.
"""


class DownloadError(KenningError):
    """
    Resource could not be downloaded due to a network error or another unknown
    error.
    """

    pass


class ChecksumVerifyError(KenningError):
    """
    Exception raised when downloaded file has invalid checksum.
    """

    pass


class ArgsManagerConvertError(KenningError):
    """
    JSON arguments given to Kenning could not be parsed or converted to a valid
    format.
    """

    pass


class ModelTooLargeError(KenningError):
    """
    Optimization flow could not be executed, because the model is too large and
    will not fit into the memory of the selected device.
    """

    pass


"""
Errors related to configuration and compatibility of different Kenning modules,
as well as the general process of building a pipeline from Kenning modules.
"""


class ConfigurationError(KenningError):
    """
    Error raised, when attempting to use unsupported configuration.
    """


class IOCompatibilityError(ConfigurationError):
    """
    Attempted to use Kenning modules with incompatible input/output.
    """

    pass


class IOSpecNotFound(ConfigurationError):
    """
    Kenning module IO specification not found.
    """

    pass


class NotSupportedError(ConfigurationError):
    """
    Raised when attempting to use a functionality, that is not supported by
    Kenning (or by that specific module implementation).
    """

    pass


class IOSpecWrongFormat(ConfigurationError):
    """
    Kenning module IO specification contains unsupported entries.
    """

    pass


class ClassInfoInvalidArgument(ConfigurationError):
    """
    Could not retrieve information on the given class, because arguments were
    invalid.
    """

    pass


class AmbiguousImport(ConfigurationError):
    """
    Exception raised by ClassLoader if two or more classes with a provided name
    exist.
    """

    pass


"""
Errors raised in implementations of the kenning.core.modelwrapper module.
"""


class KenningModelWrapperError(KenningError):
    """
    Generic error in the ModelWrapper module.
    """


class TrainingParametersMissingError(KenningModelWrapperError):
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


"""
Errors specific to the AutoML workflow.
"""


class KenningAutoMLError(KenningError):
    """
    Generic error in the AutoML module.
    """

    pass


class ModelExtractionError(KenningAutoMLError):
    """
    Raised when the model could not be properly extracted from ModelWrapper.
    """

    pass


class ModelClassNotValid(KenningAutoMLError):
    """
    Raised when provided model class cannot be imported.
    """

    pass


class AutoMLInvalidSchemaError(KenningAutoMLError):
    """
    Raised when `arguments_structure` contains not enough information
    or when data are invalid.
    """

    pass


class AutoMLInvalidArgumentsError(KenningAutoMLError):
    """
    Raised when provided arguments (in `use_model`) do not match with
    model wrapper `arguments_structure`.
    """

    pass


class AutoMLModelSizeError(KenningAutoMLError):
    """
    Raised when the model size is invalid.
    """

    def __init__(self, model_size: float, *args):
        super().__init__(*args)
        self.model_size = model_size


"""
Errors raised by implementations of the kenning.core.dataconverter module.
"""


class KenningDataConverterError(KenningError):
    """
    Generic error in the DataConverter module.
    """

    pass


"""
Errors raised by implementations of the kenning.core.dataprovider module.
"""


class KenningDataProviderError(KenningError):
    """
    Generic error in the DataProvider module.
    """

    pass


class InputDeviceError(KenningDataProviderError):
    """
    Exception to be raised when fetching data from a device fails.
    """

    pass


"""
Errors raised by implementations of the kenning.core.dataset module.
"""


class KenningDatasetError(KenningError):
    """
    Generic error in the Dataset module.
    """

    pass


class CannotDownloadDatasetError(KenningDatasetError):
    """
    Selected dataset cannot be automatically downloaded.
    """

    pass


"""
Errors raised by implementations of the kenning.core.onnxconversion module,
related to the process of converting models in ONNX format to other formats.
"""


class KenningONNXConverterError(KenningError):
    """
    Generic error in the ONNX converter module.
    """

    pass


"""
Errors raised by implementations of kenning.core.optimizer module.
"""


class KenningOptimizerError(KenningError):
    """
    Generic error in the Optimizer module.
    """

    pass


class EdgeTPUCompilerError(KenningOptimizerError):
    """
    edgetpu_compiler failed to compile the model.
    """

    pass


class ConversionError(KenningOptimizerError):
    """
    Model conversion faled.
    """

    pass


class CompilationError(KenningOptimizerError):
    """
    Compilation process failed.
    """

    pass


class IOSpecificationNotFoundError(KenningOptimizerError):
    """
    Could not find model IO specification (IOSpec).
    """

    pass


class OptimizedModelSizeError(KenningOptimizerError):
    """
    Could not retrieve size of the model to optimize.
    """

    pass


"""
Errors raised by implementations of the kenning.core.outputcollector module.
"""


class KenningOutputCollectorError(KenningError):
    """
    Generic error in the OutputCorrector module.
    """

    pass


"""
Errors raised by implementations of the kenning.core.platform module.
"""


class KenningPlatformError(KenningError):
    """
    Generic error in the Platform module.
    """

    pass


class UARTNotFoundInDTSError(KenningPlatformError):
    """
    Using runtime working on Zephyr RTOS and UART Protocol, but no UART port
    could be found in the Zephyr device tree.
    """

    pass


class RenodeSimulationError(KenningPlatformError):
    """
    Exception raised when a Renode command fails.
    """

    pass


"""
Errors thrown by implementations of the kenning.core.protocol module.
"""


class KenningProtocolError(KenningError):
    """
    Generic error in the Protocol module.
    """

    pass


class ProtocolNotStartedError(KenningProtocolError):
    """
    Exception raised by the Protocol, when attempting to use a protocol
    object, that is not initialized with 'initialize_client' or
    'initialize_server' methods.
    """

    pass


class RequestFailure(KenningProtocolError):
    """
    Request sent to the inference server running on a remote target device has
    failed.
    """

    pass


"""
Errors raised by implementations of the kenning.core.runtimebuilder module,
related to the process of automatic building of external runtimes.
"""


class KenningRuntimeBuilderError(KenningError):
    """
    Generic error in the RuntimeBuilder module.
    """

    pass


"""
Errors related to the Visual Editor (Kenning Pipeline Manager).
"""


class VisualEditorError(KenningError):
    """
    Generic error related to the Visual Editor.
    """


class VisualEditorGraphParserError(VisualEditorError):
    """
    When using the Visual Editor, an error occurred either when
    parsing a graph from the editor to a scenario, or when trying to render
    the scenario as a graph.
    """

    pass


"""
Errors raised by implementations of kenning.core.runtime module, related to the
process of model inference.
"""


class KenningRuntimeError(KenningError):
    """
    Generic error in the Runtime module (module running inference on models).
    """

    pass


class ModelNotPreparedError(KenningRuntimeError):
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


class InputNotPreparedError(KenningRuntimeError):
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


class ModelNotLoadedError(KenningRuntimeError):
    """Exception raised if a model could not be loaded."""

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


class Ai8xIzerError(KenningError):
    """
    Raised when ai8xizer.py script fails.
    """

    def __init__(self, model_size: Optional[float] = None, *args):
        super().__init__(*args)
        self.model_size = model_size
