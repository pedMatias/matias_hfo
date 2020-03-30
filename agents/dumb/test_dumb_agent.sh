#!/bin/bash
killall -9 rcssserver

export PYTHONPATH="/home/matias/Desktop/HFO"

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python

echo $HFO
$HFO --offense-agents 1 --offense-npcs 0 --defense-npcs 1 \
 --offense-on-ball 1  --trials 1 --fullstate --no-sync &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

OFFENSE_AGENT_FILE=$BASE_DIR/matias_hfo/agents/dumb/dumb_agent.py

sleep 5
$PYTHON $OFFENSE_AGENT_FILE &
# .py &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
#trap "kill -TERM -$$" SIGINT
#wait

