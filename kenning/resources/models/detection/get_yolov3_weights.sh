#!/bin/sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

wget https://pjreddie.com/media/files/yolov3.weights -O $SCRIPT_DIR/yolov3.weights
