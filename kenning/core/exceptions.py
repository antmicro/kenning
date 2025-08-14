# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0
"""
All custom exceptions thrown in Kenning.
"""


class KenningError(Exception):
    """
    Generic error in Kenning.
    """


"""
Miscellaneous errors.
"""


class DownloadError(KenningError):
    """
    Resource could not be downloaded due to a network error or another unknown
    error.
    """


class ChecksumVerifyError(KenningError):
    """
    Exception raised when downloaded file has invalid checksum.
    """


class ArgsManagerConvertError(KenningError):
    """
    JSON arguments given to Kenning could not be parsed or converted to a valid
    format.
    """


class ModelTooLargeError(KenningError):
    """
    Optimization flow could not be executed, because the model is too large and
    will not fit into the memory of the selected device.
    """


"""
Errors related to configuration and compatibility of different Kenning modules,
as well as the general process of building a pipeline from Kenning modules.
"""


class ConfigurationError(KenningError):
    """
    Error raised, when attempting to use unsupported configuration.
    """


class ModulesIncompatibleError(ConfigurationError):
    """
    Attempted to use Kenning modules with incompatible input/output.
    """


class ModuleIOSpecificationNotFoundError(ConfigurationError):
    """
    Kenning module IO specification not found.
    """


class NotSupportedError(ConfigurationError):
    """
    Raised when attempting to use a functionality, that is not supported by
    Kenning (or by that specific module implementation).
    """


class ModuleIOSpecificationFormatError(ConfigurationError):
    """
    Kenning module IO specification contains unsupported entries.
    """


class ClassInfoInvalidArgumentError(ConfigurationError):
    """
    Could not retrieve information on the given class, because arguments were
    invalid.
    """


class AmbiguousModuleError(ConfigurationError):
    """
    Exception raised by ClassLoader if two or more classes with a provided name
    exist.
    """


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


class ModelExtractionError(KenningAutoMLError):
    """
    Raised when the model could not be properly extracted from ModelWrapper.
    """


class ModelClassNotValidError(KenningAutoMLError):
    """
    Raised when provided model class cannot be imported.
    """


class InvalidSchemaError(KenningAutoMLError):
    """
    Raised when `arguments_structure` contains not enough information
    or when data are invalid.
    """


class InvalidArgumentsError(KenningAutoMLError):
    """
    Raised when provided arguments (in `use_model`) do not match with
    model wrapper `arguments_structure`.
    """


class ModelSizeError(KenningAutoMLError):
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


"""
Errors raised by implementations of the kenning.core.dataprovider module.
"""


class KenningDataProviderError(KenningError):
    """
    Generic error in the DataProvider module.
    """


class InputDeviceError(KenningDataProviderError):
    """
    Exception to be raised when fetching data from a device fails.
    """


"""
Errors raised by implementations of the kenning.core.dataset module.
"""


class KenningDatasetError(KenningError):
    """
    Generic error in the Dataset module.
    """


class CannotDownloadDatasetError(KenningDatasetError):
    """
    Selected dataset cannot be automatically downloaded.
    """


"""
Errors raised by implementations of the kenning.core.onnxconversion module,
related to the process of converting models in ONNX format to other formats.
"""


class KenningONNXConverterError(KenningError):
    """
    Generic error in the ONNX converter module.
    """


"""
Errors raised by implementations of kenning.core.optimizer module.
"""


class KenningOptimizerError(KenningError):
    """
    Generic error in the Optimizer module.
    """


class EdgeTPUCompilerError(KenningOptimizerError):
    """
    edgetpu_compiler failed to compile the model.
    """


class ConversionError(KenningOptimizerError):
    """
    Model conversion failed.
    """


class CompilationError(KenningOptimizerError):
    """
    Compilation process failed.
    """


class IOSpecificationNotFoundError(KenningOptimizerError):
    """
    Could not find model IO specification (IOSpec).
    """


class OptimizedModelSizeError(KenningOptimizerError):
    """
    Could not retrieve size of the model to optimize.
    """


"""
Errors raised by implementations of the kenning.core.outputcollector module.
"""


class KenningOutputCollectorError(KenningError):
    """
    Generic error in the OutputCorrector module.
    """


"""
Errors raised by implementations of the kenning.core.platform module.
"""


class KenningPlatformError(KenningError):
    """
    Generic error in the Platform module.
    """


class UARTNotFoundInDTSError(KenningPlatformError):
    """
    Using runtime working on Zephyr RTOS and UART Protocol, but no UART port
    could be found in the Zephyr device tree.
    """


class RenodeSimulationError(KenningPlatformError):
    """
    Exception raised when a Renode command fails.
    """


"""
Errors thrown by implementations of the kenning.core.protocol module.
"""


class KenningProtocolError(KenningError):
    """
    Generic error in the Protocol module.
    """


class ProtocolNotStartedError(KenningProtocolError):
    """
    Exception raised by the Protocol, when attempting to use a protocol
    object, that is not initialized with 'initialize_client' or
    'initialize_server' methods.
    """


class RequestFailure(KenningProtocolError):
    """
    Request sent to the inference server running on a remote target device has
    failed.
    """


"""
Errors raised by implementations of the kenning.core.runtimebuilder module,
related to the process of automatic building of external runtimes.
"""


class KenningRuntimeBuilderError(KenningError):
    """
    Generic error in the RuntimeBuilder module.
    """


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


"""
Errors raised by implementations of kenning.core.runtime module, related to the
process of model inference.
"""


class KenningRuntimeError(KenningError):
    """
    Generic error in the Runtime module (module running inference on models).
    """


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
