#!/bin/bash
killall -9 rcssserver

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
PROJECT_DIR=$BASE_DIR/matias_hfo

export PYTHONPATH=$HFO:$PYTHONPATH
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
echo $PYTHONPATH

NUM_EPISODES=20
NUM_DEFENSES=1
NUM_DEFENSES_NPCS=0
TOTAL_OPPONENTS=1

NUM_OFFENSES=1
NUM_OFFENSES_NPCS=0
TOTAL_TEAMMATES=0

OFFENSE_AGENT_FILE=$PROJECT_DIR/agents/q_agent/test_agent.py
MODEL=$PROJECT_DIR/data/qlearning_agent_4000ep_1dumbdefense.npy
DEFENSE_AGENT_FILE=$BASE_DIR/matias_hfo/agents/goalkeeper/test_agent.py

echo $HFO
$HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
 --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
 --offense-on-ball 11  --trials $NUM_EPISODES \
 --deterministic --fullstate --no-sync --no-logging &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 5
echo "Connect to player"
$PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
--num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_EPISODES \
--load_file=$MODEL &
echo "PLayer connected"

sleep 5
echo "Connect Defense Player"
$PYTHON $DEFENSE_AGENT_FILE  --num_episodes=$NUM_EPISODES &

# .py &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

