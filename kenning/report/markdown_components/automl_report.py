# Copyright (c) 2025 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Module used for Automl report generation stage.
"""


from collections import defaultdict
from importlib.resources import path
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from matplotlib.colors import to_hex

from kenning.core.drawing import Plot
from kenning.report.markdown_components.general import (
    create_report_from_measurements,
    get_plot_wildcard_path,
)
from kenning.resources import reports
from kenning.utils.logger import KLogger


def automl_report(
    automl_stats: Dict,
    imgdir: Path,
    root_dir: Path,
    image_formats: Set[str],
    colors: Optional[List] = None,
    draw_titles: bool = True,
) -> str:
    """
    Creates summary of AutoML process, containing overview
    of training process and generated models.

    Parameters
    ----------
    automl_stats : Dict
        Statistics of AutoML flow.
    imgdir : Path
        Path to the directory for images.
    root_dir : Path
        Path to the root of the documentation project
        involving this report.
    image_formats : Set[str]
        Collection with formats which should be used to
        generate plots.
    colors : Optional[List]
        Colors to be used in the plots.
    draw_titles : bool
        Should titles be drawn on the plot.

    Returns
    -------
    str
        Content of the report in MyST format.
    """
    from kenning.core.drawing import Barplot, LinePlot

    KLogger.info("Running automl_report")

    automl_data = {"general_info": automl_stats["general_info"]}
    chosen_metric = automl_stats["general_info"]["Optimized metric"]

    # Create bar plot with trained models' metrics
    trained_models_plot = imgdir / "trained_models_plot"
    if "trained_model_metrics" in automl_stats:
        models_metrics = automl_stats["trained_model_metrics"]
        models = list(models_metrics.keys())
        stages = set()
        for m in models:
            stages = stages.union(models_metrics[m].keys())
        models = list(
            filter(
                lambda x: any([s in models_metrics[x] for s in stages]),
                models,
            )
        )
        metrics = {
            k: [
                models_metrics[n].get(k, {}).get(chosen_metric, float("NaN"))
                for n in models
            ]
            for k in stages
        }
        Barplot(
            title="Metrics of models trained by AutoML flow"
            if draw_titles
            else None,
            x_label="Model ID",
            y_label=f"Optimized metric: {chosen_metric}",
            x_data=models,
            y_data=metrics,
            colors=colors,
            vertical_x_labels=False,
        ).plot(trained_models_plot, image_formats)
        automl_data["trained_models_plot"] = get_plot_wildcard_path(
            trained_models_plot, root_dir
        )

    # Create training overview
    training_plot = imgdir / "training_plot"
    comparison_training_plot = imgdir / "comparison_training_plot"
    if "training_data" in automl_stats and "training_epochs" in automl_stats:
        training_data = automl_stats["training_data"]
        training_epochs = automl_stats["training_epochs"]

        gathered = []
        for data_type in ("validation_epoch", "training_epoch"):
            # Process data to LinePlot format
            lines = defaultdict(lambda: ([], []))
            for m_id, data in training_data.items():
                if data_type not in data:
                    continue
                training_iter = 0
                epoch = training_epochs[m_id][training_iter]["epoch_range"][0]
                val_data = data[data_type]
                for _stime, loss in val_data.items():
                    # Append NaN to make gap in the line
                    if (
                        epoch
                        == training_epochs[m_id][training_iter]["epoch_range"][
                            1
                        ]
                        + 1
                        or float(_stime)
                        >= training_epochs[m_id][training_iter]["end_time"]
                    ):
                        lines[m_id][0].append(float("NaN"))
                        lines[m_id][1].append(0.0)
                        training_iter += 1
                        if training_iter >= len(training_epochs[m_id]):
                            KLogger.warning(
                                "The epoch of AutoML training "
                                "plot exceeds the range reported "
                                "by the flow, "
                                "further data from this model "
                                "will be skipped."
                            )
                            break
                        epoch = training_epochs[m_id][training_iter][
                            "epoch_range"
                        ][0]

                    lines[m_id][0].append(epoch)
                    lines[m_id][1].append(loss)
                    epoch += 1
            gathered.append(lines)

        # Sort IDs in ascending order and convert back to strings
        model_ids = set(gathered[0].keys()) | set(gathered[1].keys())
        model_ids = list(map(str, sorted(map(int, model_ids))))

        if len(colors) < len(model_ids):
            colors += [
                to_hex(c)
                for c in Plot._get_comparison_color_scheme(
                    len(model_ids) - len(colors)
                )
            ]

        LinePlot(
            lines=sum(
                ([g[i] for g in gathered if i in g] for i in model_ids),
                start=[],
            ),
            x_label="Epoch",
            y_label="Loss",
            title="Loss across AutoML flow" if draw_titles else None,
            colors=sum(
                (
                    [colors[i] for g in gathered if _id in g]
                    for i, _id in enumerate(model_ids)
                ),
                start=[],
            ),
            y_scale="log",
            lines_labels=sum(
                (
                    [
                        f"Model {i} ({'val' if j == 0 else 'train'})"
                        for j, g in enumerate(gathered)
                        if i in g
                    ]
                    for i in model_ids
                ),
                start=[],
            ),
            dashed=sum(
                (
                    [j != 0 for j, g in enumerate(gathered) if i in g]
                    for i in model_ids
                ),
                start=[],
            ),
            add_points=True,
        ).plot(training_plot, image_formats)
        automl_data["training_plot"] = get_plot_wildcard_path(
            training_plot, root_dir
        )

        # Shift data to start each training from 0th epoch
        for g in gathered:
            for m_id, lines in g.items():
                g[m_id] = list(
                    zip(*[(x, y) for x, y in zip(*lines) if not np.isnan(x)])
                )
                g[m_id][0] = tuple(range(len(g[m_id][0])))

        lines = gathered[0]
        LinePlot(
            lines=[lines[i] for i in model_ids if i in lines],
            x_label="Epoch",
            y_label="Validation loss",
            title="Comparison of models validation losses"
            if draw_titles
            else None,
            colors=[
                colors[i] for i, _id in enumerate(model_ids) if _id in lines
            ],
            y_scale="log",
            lines_labels=[f"Model {i}" for i in model_ids if i in lines],
            add_points=True,
        ).plot(comparison_training_plot, image_formats)
        automl_data["comparison_training_plot"] = get_plot_wildcard_path(
            comparison_training_plot, root_dir
        )

    # Prepare data for table with model parameters
    if "model_params" in automl_stats and (
        model_params := automl_stats["model_params"]
    ):
        automl_data["model_params"] = model_params
        # Sort strings with numbers in ascending order
        automl_data["model_ids"] = list(
            map(str, sorted(list(map(int, model_params.keys()))))
        )
        model_params_types = set()
        for v in model_params.values():
            model_params_types |= set(v.keys())
        automl_data["models_params_types"] = sorted(model_params_types)

    with path(reports, "automl.md") as report_template:
        return create_report_from_measurements(report_template, automl_data)
