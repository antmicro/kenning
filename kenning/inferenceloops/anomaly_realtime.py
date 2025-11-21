# Copyright (c) 2020-2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module containing an implementation of the anomaly inference loop.
"""
import sklearn
import sklearn.metrics

from kenning.core.measurements import Measurements
from kenning.inferenceloops.sensor_realtime import SensorRealtimeInferenceLoop


class AnomalyDetectionInferenceLoop(SensorRealtimeInferenceLoop):
    """
    Implementation of infenrece loop used for anomaly detection.
    """

    arguments_structure = {
        "smoothing_window_size": {
            "argparse_name": "--smoothing-window-size",
            "type": int,
            "default": 10,
        }
    }

    def __init__(
        self,
        dataset,
        dataconverter,
        model_wrapper,
        platform=None,
        protocol=None,
        runtime=None,
        smoothing_window_size=10,
    ):
        super().__init__(
            dataset, dataconverter, model_wrapper, platform, protocol, runtime
        )
        self.smoothing_window_size = smoothing_window_size

    def _compute_detections_and_scored_results(self, samples, results):
        """
        Matches samples with results to score all classifications
        and detections.
        """
        scored_results = []
        detections = []

        anomaly_pred_time = 0
        prev_result_idx = 0

        anomaly = False
        anomaly_time = 0

        detection_delay_counter = 0
        for sample, n_sample in zip(samples, samples[1:]):
            sample_time, s = sample
            n_sample_time, _ = n_sample

            c_anomaly = s[1][0] < s[1][1]  # e.g. [ [...], [0.0, 1.0] ]

            if anomaly != c_anomaly:
                anomaly_time = sample_time
                anomaly = c_anomaly

            for result_idx, (result, n_result) in enumerate(
                zip(results[prev_result_idx:], results[prev_result_idx + 1 :])
            ):
                result_time, r = result

                if sample_time < result_time and result_time < n_sample_time:
                    prev_result_idx = result_idx
                    anomaly_pred = r[0] == 1

                    scored_results.append(
                        {
                            "sample_time": sample_time,
                            "result_time": result_time,
                            "target": anomaly,
                            "pred": anomaly_pred,
                        }
                    )

                    if anomaly_time > anomaly_pred_time and anomaly:
                        if anomaly_pred:
                            anomaly_pred_time = result_time
                            detections.append(
                                {
                                    "target_time": anomaly_time,
                                    "prediction_time": anomaly_pred_time,
                                    "result_delay": detection_delay_counter,
                                }
                            )
                            detection_delay_counter = 0
                        else:
                            detection_delay_counter += 1

                    elif detection_delay_counter:
                        detections.append(
                            {
                                "target_time": anomaly_time,
                                "prediction_time": None,
                                "result_delay": detection_delay_counter,
                            }
                        )
                        detection_delay_counter = 0

        return scored_results, detections

    def _compute_chunks(self, scored_results, window_func):
        """
        Divides results into chunks using the target category and computes
        score using the provided window function.
        """
        cur_chunk_state = False
        cur_chunk = []
        chunks = []
        for i in range(len(scored_results)):
            n_result = scored_results[i]
            if (
                n_result["target"] != cur_chunk_state
                or i == len(scored_results) - 1
            ):
                chunks.append(
                    {
                        "target_time": n_result["sample_time"],
                        "target": cur_chunk_state,
                        "results": [*cur_chunk],
                    }
                )
                cur_chunk = []
                cur_chunk_state = n_result["target"]
                window_func.window_buffer = [
                    False
                ] * self.smoothing_window_size
            anomaly_score = window_func(n_result["pred"])
            cur_chunk.append(
                {"time": n_result["result_time"], "score": anomaly_score}
            )

        return chunks

    def _compute_metric_per_threshold(self, chunks, threshold_number=20):
        """
        Computes ADD, FDR and FAR for multiple thresholds.
        """
        metrics_per_threshold = {}
        thresholds = [
            x / threshold_number for x in range(1, threshold_number + 1)
        ]

        for threshold in thresholds:
            expected_alarms = 0
            false_alarms = 0
            detected_alarms = 0

            delays = []

            for chunk in chunks:
                target = chunk["target"]
                anomaly = False

                for i, result in enumerate(chunk["results"]):
                    if result["score"] >= threshold:
                        anomaly = True
                        if target:
                            delays.append(i)
                        break

                if target:
                    expected_alarms += 1
                if target and anomaly:
                    detected_alarms += 1
                if not target and anomaly:
                    false_alarms += 1

            metrics_per_threshold[threshold] = {
                "fdr": detected_alarms / expected_alarms
                if expected_alarms
                else 1.0,
                "far": false_alarms / (detected_alarms + false_alarms)
                if (detected_alarms + false_alarms)
                else 0.0,
                "add": sum(delays) / len(delays) if len(delays) else 0,
            }

        return metrics_per_threshold

    def _compute_confusion_f1_acc(self, scored_results):
        """
        Computes confusion matrix, F1-score and accuracy using scored results.
        """
        y_true = [x["target"] for x in scored_results]
        y_pred = [x["pred"] for x in scored_results]

        labels = [False, True]

        confusion_matrix = sklearn.metrics.confusion_matrix(
            y_true, y_pred, labels=labels
        )

        acc = sklearn.metrics.accuracy_score(y_true, y_pred)

        f1 = sklearn.metrics.f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="binary",
            pos_label=True,
        )

        return confusion_matrix, f1, acc

    def _compute_metrics(self, measurements: Measurements):
        samples = list(measurements.get_values("samples"))
        results = list(measurements.get_values("results"))

        def window_func(sample):
            if not hasattr(window_func, "window_buffer"):
                window_func.window_buffer = [
                    False
                ] * self.smoothing_window_size

            window_buffer = window_func.window_buffer

            window_buffer.append(sample)
            if len(window_buffer) > self.smoothing_window_size:
                window_buffer.pop(0)
            return sum(window_buffer) / len(window_buffer)

        (
            scored_results,
            detections,
        ) = self._compute_detections_and_scored_results(samples, results)

        chunks = self._compute_chunks(scored_results, window_func)

        metrics_per_threshold = self._compute_metric_per_threshold(chunks)

        measurements += {
            "scored_results": scored_results,
            "detections": detections,
            "metrics_per_threshold": metrics_per_threshold,
        }

        confusion_matrix, f1, acc = self._compute_confusion_f1_acc(
            scored_results
        )

        measurements += {
            "eval_confusion_matrix": confusion_matrix.tolist(),
            "class_names": ["normal", "anomaly"],
        }

        measurements += {
            "anomaly_metrics": {
                "accuracy": acc,
                "f1": f1,
            }
        }
