"""
Provides a runner that performs inference.
"""

from typing import Dict, List, Tuple, Any
from copy import deepcopy

from kenning.core.dataset import Dataset
from kenning.core.model import ModelWrapper
from kenning.core.runtime import Runtime
from kenning.core.runner import Runner
from kenning.utils.args_manager import get_parsed_json_dict
from kenning.utils.class_loader import load_class


class ModelRuntimeRunner(Runner):
    """
    Runner that performs inference using given model and runtime.
    """

    def __init__(
            self,
            model: ModelWrapper,
            runtime: Runtime,
            inputs_sources: Dict[str, Tuple[int, str]] = {},
            inputs_specs: Dict[str, Dict] = {},
            outputs: Dict[str, str] = {}):
        """
        Creates the model runner.

        Parameters
        ----------
        model : ModelWrapper
            Selected model
        runtime : Runtime
            Runtime used to run selected model
        inputs_sources : Dict[str, Tuple[int, str]]
            Input from where data is being retrieved
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs
        outputs : Dict[str, str]
            Outputs of this Runner
        """
        self.model = model
        self.runtime = runtime

        self.runtime.inference_session_start()
        self.runtime.prepare_local()

        super().__init__(
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs
        )

    def cleanup(self):
        self.runtime.inference_session_end()

    @classmethod
    def _form_parameterschema(cls):
        """
        Wrapper for creating parameterschema structure
        for the ModelRuntimeRunner class.

        Returns
        -------
        Dict :
            schema for the class
        """
        parameterschema = {
            "type": "object",
            "properties": {
                "model_wrapper": {
                    "type": "object",
                    "real_name": "model_wrapper"
                },
                "dataset": {
                    "type": "object",
                    "real_name": "dataset"
                },
                "runtime": {
                    "type": "object",
                    "real_name": "runtime"
                }
            },
            "required": ["model_wrapper", "runtime"],
            "additionalProperties": False
        }

        return parameterschema

    @classmethod
    def form_parameterschema(cls):
        """
        Creates schema for the ModelRuntimeRunner class.

        Returns
        -------
        Dict :
            schema for the class
        """
        parameterschema = cls._form_parameterschema()

        return parameterschema

    @classmethod
    def from_json(
            cls,
            json_dict: Dict,
            inputs_sources: Dict[str, Tuple[int, str]],
            inputs_specs: Dict[str, Dict],
            outputs: Dict[str, str]):
        """
        Constructor wrapper that takes the parameters from json dict.

        This function checks if the given dictionary is valid according
        to the json schema defined.
        If it is then it invokes the constructor.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor
        inputs_sources : Dict[str, Tuple[int, str]]
            Input from where data is being retrieved
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs
        outputs : Dict[str, str]
            Outputs of this Runner

        Returns
        -------
        ModelRuntimeRunner :
            object of class ModelRuntimeRunner
        """

        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        model_json_dict = parsed_json_dict['model_wrapper']
        runtime_json_dict = parsed_json_dict['runtime']

        if 'dataset' in parsed_json_dict.keys():
            dataset = cls._create_dataset(parsed_json_dict['dataset'])
        else:
            dataset = None

        model: ModelWrapper = cls._create_model(dataset, model_json_dict)
        model.prepare_model()
        runtime: Runtime = cls._create_runtime(runtime_json_dict)

        return cls(
            model,
            runtime,
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs
        )

    @staticmethod
    def _create_dataset(json_dict: Dict):
        """
        Method used to create dataset based on json dict.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor

        -------
        Dataset :
            Created dataset
        """
        cls = load_class(json_dict['type'])
        return cls.from_json(
            json_dict=json_dict['parameters'])

    # TODO: make dataset/protocol an optional parameter for model/runtime
    @staticmethod
    def _create_model(dataset: Dataset, json_dict: Dict):
        """
        Method used to create model based on json dict.

        Parameters
        ----------
        dataset : Dataset
            Dataset used to initialize model params (class names etc.)
        json_dict : Dict
            Arguments for the constructor

        -------
        ModelWrapper :
            Created model
        """
        cls = load_class(json_dict['type'])
        return cls.from_json(
            dataset=dataset,
            json_dict=json_dict['parameters'])

    @staticmethod
    def _create_runtime(json_dict):
        """
        Method used to create runtime based on json dict.

        Parameters
        ----------
        json_dict : Dict
            Arguments for the constructor

        -------
        Runtime :
            Created runtime
        """
        cls = load_class(json_dict['type'])
        return cls.from_json(
            protocol=None,
            json_dict=json_dict['parameters'])

    @classmethod
    def _get_io_specification(cls, model: ModelWrapper):
        """
        Creates runner IO specification from chosen parameters

        Parameters
        ----------
        model : ModelWrapper
            Argument for `ModelRuntimeRunner` constructor

        Returns
        -------
        Dict[str, List[Dict]] :
            Dictionary that conveys input and output layers specification
        """
        model_io_spec = model.get_io_specification()
        for io in ('input', 'output'):
            if f'processed_{io}' not in model_io_spec.keys():
                model_io_spec[f'processed_{io}'] = []
                for spec in model_io_spec[io]:
                    spec = deepcopy(spec)
                    spec['name'] = 'processed_' + spec['name']
                model_io_spec[f'processed_{io}'].append(spec)

        return model_io_spec

    @classmethod
    def parse_io_specification_from_json(cls, json_dict):
        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)

        model_json_dict = parsed_json_dict['model_wrapper']
        if 'dataset' in parsed_json_dict.keys():
            dataset = cls._create_dataset(parsed_json_dict['dataset'])
        else:
            dataset = None

        model = cls._create_model(dataset, model_json_dict)
        return cls._get_io_specification(model)

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        return self._get_io_specification(self.model)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        model_input = inputs.get('processed_input')
        if model_input is None:
            model_input = inputs['input']

        preds = self.runtime.infer(
            model_input,
            self.model,
            postprocess=False
        )
        posty = self.model.postprocess_outputs(preds)

        io_spec = self.get_io_specification()

        result = {}
        # TODO: Add support for multiple inputs/outputs
        for out_spec, out_value in zip(io_spec['output'], [preds]):
            result[out_spec['name']] = out_value

        for out_spec, out_value in zip(io_spec['processed_output'], [posty]):
            result[out_spec['name']] = out_value

        return result
