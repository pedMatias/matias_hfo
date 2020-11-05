#!/bin/bash
killall -9 rcssserver

PORT=6000

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
MODELS_DIR=$BASE_DIR/matias_hfo/models
AGENTS_DIR=$BASE_DIR/matias_hfo/plastic_agent

echo "PYTHON: ${PYTHON}"

export PYTHONPATH=$BASE_DIR:$PYTHONPATH
export PYTHONPATH=$BASE_DIR/matias_hfo:$PYTHONPATH

SET_UP_FILE=$AGENTS_DIR/plastic/set_up/set_up_policies.py

TEAMMATE_NAME="gliders"

$PYTHON $SET_UP_FILE --models_dir=$MODELS_DIR --team_name=$TEAMMATE_NAME &

trap "kill -TERM -$$" SIGINT
wait

