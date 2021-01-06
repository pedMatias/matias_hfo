#!/bin/bash

export PYTHONPATH=/home/matias/Desktop/HFO:$PYTHONPATH
export PYTHONPATH=/home/matias/Desktop/HFO/matias_hfo:$PYTHONPATH
echo $PYTHONPATH

BASE_DIR=/home/matias/Desktop/HFO/matias_hfo
DIR=$BASE_DIR/data/new_dqn_10600ep_1op_2020-06-08_00:18:00

EXPORT_SCRIPT=$BASE_DIR/utils/export_metrics_plots.py

python $EXPORT_SCRIPT --dir="$DIR"

