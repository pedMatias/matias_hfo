#!/bin/bash
export PYTHONPATH=/home/matias/Desktop/HFO:$PYTHONPATH
export PYTHONPATH=/home/matias/Desktop/HFO/matias_hfo:$PYTHONPATH
echo $PYTHONPATH

BASE_DIR=/home/matias/Desktop/HFO
PYTHON=$BASE_DIR/venv/bin/python
AGENTS_DIR=$BASE_DIR/matias_hfo/agents
MODELS_DIR=$BASE_DIR/matias_hfo/models

TEAM_NAME="helios"

TRAIN_FILE=$AGENTS_DIR/plastic_dqn_v1/train_plastic/train_plastic.py
$PYTHON $TRAIN_FILE --team_name=$TEAM_NAME --dir=$MODELS_DIR &

trap "kill -TERM -$$" SIGINT
wait