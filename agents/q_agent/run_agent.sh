#!/bin/bash
killall -9 rcssserver

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
PROJECT_DIR=$BASE_DIR/matias_hfo

export PYTHONPATH=$HFO:$PYTHONPATH
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
echo $PYTHONPATH

NUM_EPISODES=10
NUM_OPPONENTS=0
NUM_TEAMMATES=0

OFFENSE_AGENT_FILE=$PROJECT_DIR/agents/q_agent/player_agent.py
MODEL=$PROJECT_DIR/data/qlearning_agent_1000_192_3_0.1.npy

echo $HFO
$HFO --offense-agents 1 --offense-npcs $NUM_TEAMMATES \
 --defense-npcs $NUM_OPPONENTS --offense-on-ball 11  --trials $NUM_EPISODES \
 --deterministic --fullstate --no-sync --no-logging &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 5
echo "Connect to player"
$PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$NUM_OPPONENTS \
--num_teammates=$NUM_TEAMMATES --num_episodes=$NUM_EPISODES \
--load_file=$MODEL &
echo "PLayer connected"
# .py &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

