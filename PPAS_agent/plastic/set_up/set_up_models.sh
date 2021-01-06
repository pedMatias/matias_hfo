#!/bin/bash
PORT=6000

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
MODELS_DIR=$BASE_DIR/matias_hfo/models
AGENTS_DIR=$BASE_DIR/matias_hfo/multi_agents

echo "PYTHON: ${PYTHON}"

export PYTHONPATH=$BASE_DIR:$PYTHONPATH
export PYTHONPATH=$BASE_DIR/matias_hfo:$PYTHONPATH

SET_UP_FILE=$AGENTS_DIR/plastic/set_up/set_up_policies.py

GAME_SET_UP="2vs2"
TEAMMATE_NAME="aut"
DIR_NAME="${MODELS_DIR}/${GAME_SET_UP}/${TEAMMATE_NAME}"

$PYTHON $SET_UP_FILE --models_dir=$DIR_NAME --team_name=$TEAMMATE_NAME &

trap "kill -TERM -$$" SIGINT
wait

