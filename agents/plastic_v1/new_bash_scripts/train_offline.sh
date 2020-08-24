#!/bin/bash
killall -9 rcssserver

export PYTHONPATH=/home/matias/Desktop/HFO:$PYTHONPATH
export PYTHONPATH=/home/matias/Desktop/HFO/matias_hfo:$PYTHONPATH
echo $PYTHONPATH

BASE_DIR=/home/matias/Desktop/HFO
PYTHON=$BASE_DIR/venv/bin/python
MODULE_DIR=$BASE_DIR/matias_hfo/agents
TRAIN_DATA_DIR=$BASE_DIR/matias_hfo/data

# Train config:
NUM_EPISODES=2000
echo "Episodes: $NUM_EPISODES"

NUM_DEFENSES=1
NUM_DEFENSES_NPCS=0
TOTAL_DEFENSES=$(($NUM_DEFENSES + $NUM_DEFENSES_NPCS))
TOTAL_OPPONENTS=$TOTAL_DEFENSES
echo "TOTAL_OPPONENTS: $TOTAL_OPPONENTS"

NUM_OFFENSES=2
NUM_OFFENSES_NPCS=0
TOTAL_OFFENSES=$(($NUM_OFFENSES + $NUM_OFFENSES_NPCS))
TOTAL_TEAMMATES=$(($TOTAL_OFFENSES - 1))
echo "TOTAL_TEAMMATES: $TOTAL_TEAMMATES"

TRAIN_SCRIPT=$MODULE_DIR/plastic_v1/train_offline.py
DIR=$TRAIN_DATA_DIR/exploration_simplex_2000ep_1op_2020-06-28_00:05:00

sleep 2
echo "Start Train"
$PYTHON $TRAIN_SCRIPT --num_opponents=$TOTAL_OPPONENTS \
--num_teammates=$TOTAL_TEAMMATES --dir=$DIR &
echo "PLayer connected"


