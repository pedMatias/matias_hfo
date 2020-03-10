#!/bin/bash
killall -9 rcssserver

export PYTHONPATH=/home/matias/Desktop/HFO:$PYTHONPATH
export PYTHONPATH=/home/matias/Desktop/HFO/matias_hfo:$PYTHONPATH
echo $PYTHONPATH

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python

NUM_EPISODES=1000
NUM_OPPONENTS=0
NUM_TEAMMATES=0

OFFENSE_AGENT_FILE=$BASE_DIR/matias_project/agents/q_agent/learning_agent.py

echo $HFO
$HFO --offense-agents 1 --offense-npcs $NUM_TEAMMATES \
 --defense-npcs $NUM_OPPONENTS --offense-on-ball 11  --trials $NUM_EPISODES \
 --deterministic --fullstate --no-logging --headless &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)


sleep 5
echo "Connect to player"
$PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$NUM_OPPONENTS \
--num_teammates=$NUM_TEAMMATES --num_episodes=$NUM_EPISODES &
echo "PLayer connected"
# .py &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

