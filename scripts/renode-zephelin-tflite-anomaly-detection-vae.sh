#!/bin/bash

# Copyright (c) 2026 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

# This script will generate a performance report with tracing section using Zephelin.
# Requires 1 environ: ZEPHYR_WORKSPACE - with path to west workspace with Kenning Zephyr Runtime configured.
# The script takes 4 arguments: model path, dataset file path, report path and report name.
# All given paths should be relative to $ZEPHYR_WORKSPACE/kenning-zephyr-runtime directory.

set -e

cd $ZEPHYR_WORKSPACE/kenning-zephyr-runtime

west build -p -b stm32f746g_disco app -- -DEXTRA_CONF_FILE="tflite.conf;$(realpath ./zpl.conf);zpl_uart.conf" -DCONFIG_KENNING_MODEL_PATH=\"$(realpath $1)\"
west build -t board-repl

SCENARIO_PATH=/tmp/scenario-vae-tflite.yml

echo 'platform:' > $SCENARIO_PATH
echo '  type: ZephyrPlatform' >> $SCENARIO_PATH
echo '  parameters:' >> $SCENARIO_PATH
echo '    name: stm32f746g_disco' >> $SCENARIO_PATH
echo '    simulated: true' >> $SCENARIO_PATH
echo '    enable_zephelin: true' >> $SCENARIO_PATH
echo '    zephyr_build_path: ./build/' >> $SCENARIO_PATH
echo 'model_wrapper:' >> $SCENARIO_PATH
echo '  type: PyTorchAnomalyDetectionVAE' >> $SCENARIO_PATH
echo '  parameters:' >> $SCENARIO_PATH
echo '    model_path: _' >> $SCENARIO_PATH
echo 'dataset:' >> $SCENARIO_PATH
echo '  type: AnomalyDetectionDataset' >> $SCENARIO_PATH
echo '  parameters:' >> $SCENARIO_PATH
echo '    dataset_root: ./workspace/DATA' >> $SCENARIO_PATH
echo "    csv_file: $2" >> $SCENARIO_PATH
echo '    split_fraction_test: 0.1' >> $SCENARIO_PATH
echo '    split_seed: 12345' >> $SCENARIO_PATH
echo '    inference_batch_size: 1' >> $SCENARIO_PATH
echo 'optimizers:' >> $SCENARIO_PATH
echo '- type: TFLiteCompiler' >> $SCENARIO_PATH
echo '  parameters:' >> $SCENARIO_PATH
echo '    target: default' >> $SCENARIO_PATH
echo "    compiled_model_path: $1" >> $SCENARIO_PATH
echo '    inference_input_type: float32' >> $SCENARIO_PATH
echo '    inference_output_type: float32' >> $SCENARIO_PATH

kenning test report --cfg $SCENARIO_PATH --measurements results.json --report-types zephyr_traces --report-path $3 --report-name "$4" --to-html
