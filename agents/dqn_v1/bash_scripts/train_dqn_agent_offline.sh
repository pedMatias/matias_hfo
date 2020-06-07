#!/bin/bash
killall -9 rcssserver

export PYTHONPATH=/home/matias/Desktop/HFO:$PYTHONPATH
export PYTHONPATH=/home/matias/Desktop/HFO/matias_hfo:$PYTHONPATH
echo $PYTHONPATH

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
MODULE_DIR=$BASE_DIR/matias_hfo/agents


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

DEFENSE_AGENT_FILE=$MODULE_DIR/fixed_agents/goalkeeper/player_agent.py
# DEFENSE_AGENT_FILE=$MODULE_DIR/fixed_agents/goalkeeper/goalkeeper_v2.py
STATIC_AGENT_FILE=$MODULE_DIR/fixed_agents/fixed_teammate/static_agent.py

OFFENSE_AGENT_FILE=$MODULE_DIR/dqn_v1/dqn_train.py

BATCH_TRAIN_FILE=$BASE_DIR/matias_hfo/data/new_11000_1ep_2020-06-03_20:54:00/game_transictions.npy


echo "Connect to Main player"
$PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
--num_teammates=$TOTAL_TEAMMATES --train_online=$false \
--batch_train_file=$BATCH_TRAIN_FILE
echo "PLayer connected"


