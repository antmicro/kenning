# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Classes visualizing in real time outputs of classification, detection
and instance segmentation models.
"""
from typing import Dict, Tuple, List, Any
from collections import defaultdict
from copy import deepcopy
import dearpygui.dearpygui as dpg
import multiprocessing as mp
import numpy as np
import cv2
import colorsys
import threading
from scipy.special import softmax

from kenning.utils.args_manager import get_parsed_json_dict
from kenning.core.outputcollector import OutputCollector
from kenning.datasets.helpers.detection_and_segmentation import DetectObject
from kenning.datasets.helpers.detection_and_segmentation import SegmObject

_FONT_SCALE = 1.5
_FONT_SIZE = 16
_PADDING = 8
_SIDE_PANEL_WIDTH = 512
_SCORE_COLUMN_WIDTH = 80


class BaseRealTimeVisualizer(OutputCollector):
    """
    A base class for OpenGL-based real time visualizer.
    """

    setup_gui_lock = threading.Lock()

    arguments_structure = {
        'viewer_width': {
            'argparse_name': '--viewer-width',
            'description': 'Width of the visualizer window',
            'type': int,
            'default': 1280
        },
        'viewer_height': {
            'argparse_name': '--viewer-height',
            'description': 'Height of the visualizer window',
            'type': int,
            'default': 800
        },
        'input_color_format': {
            'argparse_name': '--input-color-format',
            'description': 'Color format of provided frame (RGB or BGR)',
            'type': str,
            'default': 'BGR'
        },
        'input_memory_layout': {
            'argparse_name': '--input-memory-layout',
            'description': 'Memory layout of provided frame (NCHW or NHWC)',
            'type': str,
            'default': 'NHWC'
        }
    }

    def __init__(
            self,
            title: str,
            viewer_width: int = 1280,
            viewer_height: int = 800,
            input_color_format: str = 'BGR',
            input_memory_layout: str = 'NHWC',
            inputs_sources: Dict[str, Tuple[int, str]] = {},
            inputs_specs: Dict[str, Dict] = {},
            outputs: Dict[str, str] = {}):
        """
        Base class for OpenGL-based real time visualizer.

        Parameters
        ----------
        title: str
            Name of the window.
        width: int
            Width of the window.
        height: int
            Height of the window.
        input_color_format : str
            Color format of provided frame (BGR or RGB).
        input_memory_layout : str
            Memory layout of provided frame (NCHW or NHWC).
        input_sources : Dict[str, Tuple[int, str]]
            Input from where data is being retrieved.
        inputs_specs : Dict[str, Dict]
            Specifications of runner's inputs.
        outputs : Dict[str, str]
            Outputs of this Runner.
        """
        self.title = title
        self.width = viewer_width
        self.height = viewer_height
        self.input_color_format = input_color_format
        self.input_memory_layout = input_memory_layout
        self.draw_layer_index = 0

        # dict containing colors assigned to classes
        # new color is created using random hue and maximum
        # saturation, value and alpha
        self.class_colors = defaultdict(
            lambda: [*colorsys.hsv_to_rgb(np.random.rand(), 1, 1), 1.]
        )

        self.stop = False
        self.process_data = mp.Queue()
        self.process = mp.Process(
            target=BaseRealTimeVisualizer._gui_thread,
            args=(self,)
        )
        self.process.start()

        super().__init__(
            inputs_sources=inputs_sources,
            inputs_specs=inputs_specs,
            outputs=outputs
        )

    @classmethod
    def _get_io_specification(cls, input_memory_layout):
        """
        Creates runner IO specification from chosen parameters.

        Parameters
        ---------
        input_memory_layout : str
            Constructor argument.

        Returns
        -------
        Dict[str, List[Dict]] :
            Dictionary that conveys input and output layers specification.
        """
        raise NotImplementedError

    def get_io_specification(self) -> Dict[str, List[Dict]]:
        return self._get_io_specification(self.input_memory_layout)

    @classmethod
    def parse_io_specification_from_json(cls, json_dict):
        parameterschema = cls.form_parameterschema()
        parsed_json_dict = get_parsed_json_dict(parameterschema, json_dict)
        return cls._get_io_specification(
            parsed_json_dict["input_memory_layout"])

    def cleanup(self):
        self.process_data.close()
        self.process_data.join_thread()
        self.process.terminate()
        self.process.join()

    def setup_gui(self):
        """
        Method that sets up GUI of the visualizer.
        """
        # if there are more than 1 visualizer we need to assure that there
        # will not be tag conflicts
        BaseRealTimeVisualizer.setup_gui_lock.acquire()
        # look for valid tag
        dpg.create_context()

        self.id = 0
        while dpg.does_item_exist(f'main_window_{self.id}'):
            self.id += 1

        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                width=self.width,
                height=self.height,
                default_value=np.zeros((self.width, self.height, 3)),
                tag=f'input_image_texture_{self.id}',
            )

        with dpg.window(
                tag=f'main_window_{self.id}',
                no_title_bar=True,
                autosize=True
        ):
            dpg.add_image(
                texture_tag=f'input_image_texture_{self.id}',
                tag=f'image_render_{self.id}',
                pos=(_PADDING, _PADDING)
            )

        dpg.set_global_font_scale(_FONT_SCALE)

        if self.id == 0:
            dpg.set_primary_window(f'main_window_{self.id}', True)
            dpg.create_viewport(
                title=self.title,
                width=self.width + _PADDING*2,
                height=self.height + _PADDING*2,
                resizable=True
            )
            dpg.setup_dearpygui()
            dpg.show_viewport()
        elif self.id == 1:
            dpg.set_primary_window('main_window_0', False)

        BaseRealTimeVisualizer.setup_gui_lock.release()

    def _gui_thread(self):
        """
        Method that performs data processing.
        Called in another process to improve performance.
        """
        self.setup_gui()

        while not self.stop:
            if dpg.is_dearpygui_running():
                data = self.process_data.get()
                self.process_output(*data)
            else:
                self.stop = True

        self.detach_from_output()

    def _draw_fps_counter(self, fps: float):
        """
        Draws FPS counter in left top corner.

        Parameters
        ----------
        fps : float
            Number of frames per second.
        """
        draw_layer_tag = f'draw_layer_{self.id}_{self.draw_layer_index^1}'

        dpg.draw_rectangle(
            parent=draw_layer_tag,
            pmin=(0, 0),
            pmax=(128, 32),
            fill=(0, 0, 0, 128),
            color=(0, 0, 0, 128)
        )

        dpg.draw_text(
            parent=draw_layer_tag,
            text=f'FPS: {fps}',
            pos=(_PADDING*2, _PADDING),
            color=(255, 255, 255, 255),
            size=_FONT_SIZE
        )

    def swap_layers(self):
        """
        Method that swaps drawing layers.
        Called when all drawings are done.
        """
        dpg.show_item(f'draw_layer_{self.id}_{self.draw_layer_index^1}')
        dpg.delete_item(f'draw_layer_{self.id}_{self.draw_layer_index}')
        self.draw_layer_index ^= 1

    def process_output(
            self,
            input_data: List[np.ndarray],
            output_data: List[Any]):
        """
        Method used to prepare data for visualization and call
        visualization method.

        Parameters
        ----------
        input_data : List[np.ndarray]
            List of input images.
        output_data : List[Any]
            List of data used for visualization.
        """
        assert len(input_data) == 1
        assert len(output_data) == 1
        img = input_data[0]
        output_data = output_data[0]

        if self.input_memory_layout == 'NCHW':
            img = img.transpose(1, 2, 0)

        if self.input_color_format == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

        img = cv2.resize(img, (self.width, self.height))

        img = self.visualize_output(img, output_data)

        dpg.set_value(f'input_image_texture_{self.id}', img)

        self._draw_fps_counter(dpg.get_frame_rate())

        if self.id == 0:
            dpg.render_dearpygui_frame()
        self.swap_layers()

    def detach_from_output(self):
        dpg.destroy_context()

    def should_close(self) -> bool:
        return not self.process.is_alive()

    def visualize_output(
            self,
            img: np.ndarray,
            output_data: Any):
        """
        Method used to visualize data.

        Parameters
        ----------
        img : np.ndarray
            Input image.
        output_data : Any
            Data used for visualization.
        """
        raise NotImplementedError

    def get_output_data(
            self,
            inputs: Dict[str, Any]) -> Any:
        """
        Retrieves data specific to visualizer from inputs.

        Parameters
        ----------
        inputs : Dict[str, Any]
            Visualized inputs.
        Returns
        -------
        Any : Data specific to visualizer.
        """
        return inputs

    def run(
            self,
            inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_data = inputs['frame']
        output_data = self.get_output_data(inputs)
        self.process_data.put((deepcopy(input_data), deepcopy(output_data)))
        return {}


class RealTimeDetectionVisualizer(BaseRealTimeVisualizer):
    """
    Visualizes output of object detection showing bounding rectangles,
    class names and scores.
    """

    def __init__(self, *args, **kwargs):
        """
        Creates the detection visualizer.
        """
        self.layer = None
        super().__init__('Real time detection visualizer', *args, **kwargs)

    @classmethod
    def _get_io_specification(cls, input_memory_layout):
        if input_memory_layout == 'NCHW':
            frame_shape = (1, 3, -1, -1)
        else:
            frame_shape = (1, -1, -1, 3)
        return {
            'input': [
                {'name': 'frame', 'shape': frame_shape, 'dtype': 'float32'},
                {'name': 'detection_data', 'type': 'List[DetectObject]'}],
            'output': []
        }

    def get_output_data(self, inputs: Dict[str, Any]) -> List[DetectObject]:
        return inputs['detection_data']

    def visualize_output(
            self,
            img: np.ndarray,
            output_data: List[DetectObject]):
        """
        Method used to visualize object detection data.

        Parameters
        ----------
        img : np.ndarray
            Input image.
        output_data : List[DetectObject]
            List of detection data.
        """
        draw_layer_tag = f'draw_layer_{self.id}_{self.draw_layer_index^1}'

        self.layer = dpg.add_draw_layer(
            parent=f'main_window_{self.id}',
            tag=draw_layer_tag,
            show=False
        )
        for out in output_data:
            description = f'{out.clsname} [{out.score*100:.2f}%]'
            # add alpha and convert to RGB
            color = np.array(self.class_colors[out.clsname])
            color = tuple((255*color).astype(np.uint8))
            dpg.draw_rectangle(
                parent=draw_layer_tag,
                pmin=(out.xmin*self.width,
                      out.ymin*self.height),
                pmax=(out.xmax*self.width,
                      out.ymax*self.height),
                color=color,
                thickness=2
            )
            dpg.draw_text(
                parent=draw_layer_tag,
                text=description,
                pos=(out.xmin*self.width + _PADDING,
                     out.ymin*self.height + _PADDING),
                color=color,
                size=_FONT_SIZE
            )

        return img


class RealTimeSegmentationVisualization(BaseRealTimeVisualizer):
    """
    Visualizes output of segmentation showing masks, class names and scores.
    """

    arguments_structure = {
        'score_threshold': {
            'argparse_name': '--score_threshold',
            'description': 'Class score threshold to be drawn.',
            'type': float,
            'default': 0.1
        }
    }

    def __init__(self, score_threshold: float, *args, **kwargs):
        """
        Creates the detection visualizer.

        Parameters
        ----------
        score_threshold : float
            Score threshold for presenting class.
        """
        self.layer = None
        self.score_threshold = score_threshold
        super().__init__('Real time segmentation visualization',
                         *args, **kwargs)

    @classmethod
    def _get_io_specification(cls, input_memory_layout):
        if input_memory_layout == 'NCHW':
            frame_shape = (1, 3, -1, -1)
        else:
            frame_shape = (1, -1, -1, 3)
        return {
            'input': [
                {'name': 'frame', 'shape': frame_shape, 'dtype': 'float32'},
                {'name': 'segmentation_data', 'type': 'List[SegmObject]'}],
            'output': []
        }

    def get_output_data(self, inputs: Dict[str, Any]) -> List[SegmObject]:
        return inputs['segmentation_data']

    def visualize_output(
            self,
            img: np.ndarray,
            output_data: List[SegmObject]):
        """
        Method used to visualize object detection data.

        Parameters
        ----------
        img : np.ndarray
            Input image.
        output_data : List[SegmObject]
            List of segmentation data.
        """
        draw_layer_tag = f'draw_layer_{self.id}_{self.draw_layer_index^1}'
        mix_factor = .3

        self.layer = dpg.add_draw_layer(
            parent=f'main_window_{self.id}',
            tag=draw_layer_tag,
            show=False
        )

        for out in output_data:
            if out.score < self.score_threshold:
                continue

            mask = cv2.resize(
                out.mask,
                (self.width, self.height),
                cv2.INTER_NEAREST
            ).astype(np.uint8)
            color_mask = np.tile(mask[..., None]/255., [1, 1, 4])
            color_image = np.tile(
                self.class_colors[out.clsname],
                [self.height, self.width, 1]
            )
            img += color_image * color_mask * mix_factor
            contour, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_NONE
            )
            color = np.array(self.class_colors[out.clsname])
            color = tuple((255*color).astype(np.uint8))
            for c in contour:
                dpg.draw_polyline(
                    parent=draw_layer_tag,
                    points=list(c[:, 0, :]),
                    color=color,
                    thickness=2
                )
                description = f'{out.clsname} [{out.score * 100:.2f}%]'
                dpg.draw_text(
                    parent=draw_layer_tag,
                    text=description,
                    pos=(c[:, 0, 0].min() + _PADDING,
                         c[:, 0, 1].min() + _PADDING),
                    color=color,
                    size=_FONT_SIZE
                )

        return img


class RealTimeClassificationVisualization(BaseRealTimeVisualizer):
    """
    Visualizes output of classification showing list of classes and scores.
    """

    arguments_structure = {
        'top_n': {
            'argparse_name': '--show-top-n',
            'description': 'Shows top N results of classification',
            'type': int,
            'default': 5
        }
    }

    def __init__(self, top_n: int = 5, *args, **kwargs):
        """
        Creates the classification visualizer.

        Parameters
        ----------
        top_n : int
            Number of classes to be listed.
        """
        super().__init__('Real time classification visualizer',
                         *args, **kwargs)
        class_input_spec = self.inputs_specs['classification_data']
        self.class_names = class_input_spec['class_names']
        self.top_n = top_n

    def setup_gui(self):
        super().setup_gui()

        dpg.add_group(
            parent=f'main_window_{self.id}',
            tag=f'side_panel_{self.id}',
            pos=(self.width + _PADDING*2, _PADDING),
            width=_SIDE_PANEL_WIDTH
        )

        with dpg.table(
            parent=f'side_panel_{self.id}',
            width=_SIDE_PANEL_WIDTH
        ):
            dpg.add_table_column(
                label='',
                init_width_or_weight=_SCORE_COLUMN_WIDTH,
                width_fixed=True
            )
            dpg.add_table_column(
                label='Score',
                init_width_or_weight=_SCORE_COLUMN_WIDTH,
                width_fixed=True
            )
            dpg.add_table_column(label='Name')

            for i in range(self.top_n):
                with dpg.table_row():
                    dpg.add_text(
                        tag=f'cell_bar_{i}_{self.id}'
                    )
                    for name in ('perc', 'name'):
                        dpg.add_text(tag=f'cell_{name}_{i}_{self.id}')

        dpg.set_viewport_width(self.width + _PADDING*2 + _SIDE_PANEL_WIDTH)

    @classmethod
    def _get_io_specification(cls, input_memory_layout):
        if input_memory_layout == 'NCHW':
            frame_shape = (1, 3, -1, -1)
        else:
            frame_shape = (1, -1, -1, 3)
        return {
            'input': [
                {'name': 'frame', 'shape': frame_shape, 'dtype': 'float32'},
                {'name': 'classification_data', 'shape': (1, -1), 'dtype': 'float32'}],  # noqa: 501
            'output': []
        }

    def get_output_data(self, inputs: Dict[str, Any]) -> np.ndarray:
        return inputs['classification_data']

    def visualize_output(
            self,
            img: np.ndarray,
            output_data: np.ndarray):
        """
        Method used to visualize object detection data.

        Parameters
        ----------
        img : np.ndarray
            Input image.
        output_data : np.ndarray
            Classification data.
        """
        draw_layer_tag = f'draw_layer_{self.id}_{self.draw_layer_index^1}'

        self.layer = dpg.add_draw_layer(
            parent=f'main_window_{self.id}',
            tag=draw_layer_tag,
            show=False
        )

        best_k = np.argsort(output_data)[-self.top_n:][::-1]
        class_names = np.array(self.class_names)[best_k]
        percentages = softmax(output_data[best_k])

        for i, (name, perc) in enumerate(zip(class_names, percentages)):
            label_pos = dpg.get_item_pos(f'cell_bar_{i}_{self.id}')
            dpg.draw_rectangle(
                parent=draw_layer_tag,
                pmin=label_pos,
                pmax=(
                    label_pos[0] + _SCORE_COLUMN_WIDTH*perc,
                    label_pos[1] + _FONT_SIZE),
                color=(0, 255, 0),
                fill=(0, 255, 0)
            )
            dpg.set_value(f'cell_perc_{i}_{self.id}', f'{perc*100:.2f}%')
            dpg.set_value(f'cell_name_{i}_{self.id}', name)

        return img
