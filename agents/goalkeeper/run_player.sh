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
NUM_OPPONENTS=1
NUM_OPPONENTS_NPCS=0
NUM_TEAMMATES=1
NUM_TEAMMATES_NPCS=0

AGENT_FILE=$PROJECT_DIR/agents/goalkeeper/player_agent.py

echo $HFO
$HFO --offense-agents $NUM_TEAMMATES --offense-npcs $NUM_TEAMMATES_NPCS \
 --defense-agents $NUM_OPPONENTS --defense-npcs $NUM_OPPONENTS_NPCS \
 --offense-on-ball 11 --trials $NUM_EPISODES \
 --deterministic --fullstate --no-sync --no-logging &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 5
echo "Connect to player"
$PYTHON $AGENT_FILE  --num_episodes=$NUM_EPISODES &
echo "PLayer connected"
# .py &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

