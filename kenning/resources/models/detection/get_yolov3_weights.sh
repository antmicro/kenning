#!/bin/sh

# Copyright (c) 2020-2023 Antmicro <www.antmicro.com>
#
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

wget https://pjreddie.com/media/files/yolov3.weights -O $SCRIPT_DIR/yolov3.weights
