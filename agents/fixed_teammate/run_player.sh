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
NUM_DEFENSES=0
NUM_DEFENSES_NPCS=0
NUM_OFFENSES=2
NUM_OFFENSES_NPCS=0

FIXED_AGENT_FILE=$PROJECT_DIR/agents/fixed_teammate/player_agent.py
STATIC_AGENT_FILE=$PROJECT_DIR/agents/fixed_teammate/static_agent.py

echo $HFO
$HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
 --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
  --trials $NUM_EPISODES \
 --deterministic --fullstate --no-sync --no-logging &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 3
echo "Connect to player"
$PYTHON $STATIC_AGENT_FILE  --num_episodes=$NUM_EPISODES --num_opponents=0 \
--num_teammates=1 &
echo "PLayer connected"

sleep 3
echo "Connect to player"
$PYTHON $FIXED_AGENT_FILE  --num_episodes=$NUM_EPISODES --num_opponents=0 \
--num_teammates=1 &
echo "PLayer connected"

# .py &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

