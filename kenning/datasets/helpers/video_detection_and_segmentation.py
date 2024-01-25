# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Base class for video object detection and segmentation datasets.
Contains common methods for video datasets.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

import cv2
import numpy as np

from kenning.core.measurements import Measurements
from kenning.datasets.helpers.detection_and_segmentation import (
    ObjectDetectionSegmentationDataset,
    compute_map_per_threshold,
)
from kenning.utils.logger import KLogger


class VideoObjectDetectionSegmentationDataset(
    ObjectDetectionSegmentationDataset, ABC
):
    """
    Base for video object detection and segmentation datasets.
    """

    @abstractmethod
    def get_recording_name(self, recording: Any) -> str:
        """
        Returns unique name of the recording within the dataset.

        Parameters
        ----------
        recording : Any
            Recording object.

        Returns
        -------
        str
            Name of the recording.
        """
        ...

    @staticmethod
    def form_subsequences(
        sequence: List[Any],
        policy: str = "subsequence",
        num_segments: int = 1,
        window_size: int = 1,
    ) -> List[List[Any]]:
        """
        Split a given sequence into subsequences according to defined `policy`.

        Parameters
        ----------
        sequence : List[Any]
            List of frames making up the sequence.
        policy : str
            The policy for splitting the sequence. Can be one of:
            * subsequence: splits the sequence into `num_segments` subsequences
            of equal length,
            * window: splits the sequence into subsequences
            of `window_size` length with stride equal to `1`. Next,
            `num_segments` subsequences are sampled from the list of
            windows on an equal stride basis.
            Defaults to subsequence.
        num_segments : int
            Number of segments to split the sequence into.
        window_size : int
            Size of the sliding window when using `sliding_window` policy.

        Returns
        -------
        List[List[Any]]
            List of subsequences.

        Raises
        ------
        ValueError
            If the policy is unknown.
        """
        if policy == "subsequence":
            segments = np.linspace(
                0,
                len(sequence),
                num_segments + 1,
                dtype=int,
                endpoint=True,
            )
            return [
                sequence[segments[i] : segments[i + 1]]
                for i in range(num_segments)
            ]
        elif policy == "window":
            windows = np.lib.stride_tricks.sliding_window_view(
                sequence, window_size
            )
            windows_indices = np.linspace(
                0,
                len(windows) - 1,
                num_segments,
                dtype=int,
                endpoint=True,
            )
            return windows[windows_indices].tolist()

        else:
            KLogger.error(f"Unknown policy: {policy}")
            raise ValueError

    @staticmethod
    def sample_items(
        segments: List[List[Any]],
        policy: str = "consecutive",
        items_per_segment: int = 1,
        seed: Optional[int] = None,
    ) -> List[Any]:
        """
        Samples items from segments into a single list of items.
        In total, `len(num_segments) * items_per_segment` items are sampled.

        For the `consecutive` policy, from each segment, a random start-index
        is sampled from which `items_per_segment` consecutive items are
        returned.

        For the `step` policy, from each segment `items_per_segment` items
        are sampled with the equal stride.

        If the number of items in the segment is lower than
        `items_per_segment`, the whole segment is returned.

        Parameters
        ----------
        segments : List[List[Any]]
            List of segments, each containing a list of items.
        policy : str
            The policy for sampling the frames. Can be one of:
            * consecutive: samples `frames_per_segment` consecutive frames
            with random start index from each segment,
            * step: samples `frames_per_segment` frames with equal stride
            from each segment.
            Defaults to consecutive.
        items_per_segment : int
            Number of items to sample from each segment.
        seed : Optional[int]
            Seed for the random number generator.

        Returns
        -------
        List[Any]
            List of sampled items.

        Raises
        ------
        ValueError
            If the policy is unknown or the number of items per segment
            is lower than 1.
        """
        if items_per_segment < 1:
            KLogger.error(
                f"Number of items per segment must be greater than 0, "
                f"got {items_per_segment}"
            )
            raise ValueError
        sampled_items = []
        if policy == "consecutive":
            if seed is not None:
                np.random.seed(seed)
            for segment in segments:
                items_length = len(segment)
                if items_length < items_per_segment:
                    sampled_items.extend(segment)
                    continue

                start_index = np.random.randint(
                    0, items_length - items_per_segment + 1
                )
                sampled_items.extend(
                    segment[start_index : start_index + items_per_segment]
                )
        elif policy == "step":
            for segment in segments:
                items_length = len(segment)
                if items_length < items_per_segment:
                    sampled_items.extend(segment)
                    continue

                item_indices = np.linspace(
                    0,
                    items_length,
                    items_per_segment,
                    dtype=int,
                    endpoint=False,
                )
                sampled_items.extend(np.array(segment)[item_indices].tolist())
        else:
            KLogger.error(f"Unknown policy: {policy}")
            raise ValueError
        return sampled_items

    def evaluate(
        self,
        predictions: List,
        truth: List,
        min_iou: float = 0.5,
        max_dets: int = 100,
    ) -> Measurements:
        """
        Evaluates the model based on the predictions.

        Computes the IoU metric for predictions and ground truth.
        Evaluation results are stored in the measurements object.

        Parameters
        ----------
        predictions : List
            The list of predictions from the model.
        truth : List
            The ground truth for given batch.
        min_iou : float
            The minimum IoU value for which the prediction is considered
            correct. Defaults to 0.5.
        max_dets : int
            The maximum number of detections to consider. Defaults to 100.

        Returns
        -------
        Measurements
            The dictionary containing the evaluation results.
        """
        measurements = Measurements()
        curr_index = self._dataindex - len(predictions)
        for sequence_preds, sequence_truth in zip(predictions, truth):
            seq_measurements = Measurements()
            seq_data = {}
            seq_data["IoU"] = []

            seq_data["name"] = self.get_recording_name(
                self.dataX[self._dataindices[curr_index]]
            )

            for preds, groundtruths in zip(sequence_preds, sequence_truth):
                frame_measurements = Measurements()
                matchedgt = np.zeros([len(groundtruths)], dtype=np.int32)
                preds.sort(key=lambda x: -x.score)
                preds = preds[:max_dets]

                # Add measurements for predictions
                for predid, pred in enumerate(preds):
                    bestiou = 0.0
                    bestgt = -1
                    for gtid, gt in enumerate(groundtruths):
                        if pred.clsname != gt.clsname:
                            continue
                        if matchedgt[gtid] > 0 and not gt.iscrowd:
                            continue
                        iou = self.compute_iou(pred, gt)
                        if iou <= bestiou:
                            continue
                        bestiou = iou
                        bestgt = gtid

                    # Don't append 0.0 IoU for gt if have any predictions
                    if bestgt != -1:
                        matchedgt[bestgt] = 1

                    frame_measurements.add_measurement(
                        f"eval_det/{pred.clsname}",
                        [
                            [
                                float(pred.score),
                                float(bestiou >= min_iou),
                                float(bestiou),
                            ]
                        ],
                        lambda: list(),
                    )

                # Add measurements for not-matched groundtruths
                for gtid, gt in enumerate(groundtruths):
                    frame_measurements.accumulate(
                        f"eval_gtcount/{gt.clsname}", 1, lambda: 0
                    )

                    if matchedgt[gtid] == 0:
                        frame_measurements.add_measurement(
                            f"eval_det/{gt.clsname}",
                            [[float(0), float(0), float(0)]],
                            lambda: list(),
                        )

                # Calculate mean IoU for each class in frame
                ious = {}
                for key in frame_measurements.data.keys():
                    if not key.startswith("eval_det"):
                        continue
                    class_name = key.split("/")[1]
                    mean_iou = np.array(
                        [x[2] for x in frame_measurements.data[key]]
                    ).mean()
                    ious[class_name] = mean_iou
                seq_data["IoU"].append(ious)

                seq_measurements += frame_measurements

            # Display frame with predictions
            if self.show_on_eval:
                self.show_eval_images(
                    sequence_preds,
                    sequence_truth,
                    self._dataindices[curr_index],
                )

            # Calculate mAP for the whole sequence
            thresholds = np.arange(0.2, 1.05, 0.05)
            maps = compute_map_per_threshold(
                {
                    "class_names": self.get_class_names(),
                    **seq_measurements.data,
                },
                thresholds,
            )

            # Total is calculated as mean of [0.5, 0.95] thresholds
            seq_data["mAP"] = {
                "total": maps[-11:][:10].mean(),
                "values": maps,
                "thresholds": thresholds,
            }

            # Update measurements
            seq_measurements.add_measurement(
                "eval/recordings",
                [seq_data],
                lambda: list(),
            )

            measurements += seq_measurements
            curr_index += 1
        return measurements

    def show_eval_images(
        self, predictions: List, truth: List, sequence_index: int
    ) -> None:
        """
        Shows the predictions on screen compared to ground truth.

        It uses proper method based on task parameter.

        The method runs a preview of inference results during
        evaluation process.

        Parameters
        ----------
        predictions : List
            The list of predictions from the model.
        truth : List
            The ground truth for given batch.
        sequence_index : int
            The index of the current sequence.

        Returns
        -------
        None
        """
        imgs = self.prepare_input_samples([self.dataX[sequence_index]])[0]
        for pred, gt, img in zip(predictions, truth, imgs):
            if self.image_memory_layout == "NCHW":
                img = img.transpose(1, 2, 0)
            int_img = np.multiply(img, 255).astype("uint8")
            int_img = cv2.cvtColor(int_img, cv2.COLOR_BGR2GRAY)
            int_img = cv2.cvtColor(int_img, cv2.COLOR_GRAY2RGB)
            int_img = self.apply_predictions(int_img, pred, gt)

            cv2.imshow("evaluated image", int_img)
            while True:
                c = cv2.waitKey(100)
                if (
                    cv2.getWindowProperty(
                        "evaluated image", cv2.WND_PROP_VISIBLE
                    )
                    < 1
                ):
                    return
                if c != -1:
                    break