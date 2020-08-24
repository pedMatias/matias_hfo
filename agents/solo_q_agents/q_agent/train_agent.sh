#!/bin/bash
killall -9 rcssserver

export PYTHONPATH=/home/matias/Desktop/HFO:$PYTHONPATH
export PYTHONPATH=/home/matias/Desktop/HFO/matias_hfo:$PYTHONPATH
echo $PYTHONPATH

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python

NUM_EPISODES=4000

NUM_DEFENSES=1
NUM_DEFENSES_NPCS=0
TOTAL_OPPONENTS=1

NUM_OFFENSES=1
NUM_OFFENSES_NPCS=0
TOTAL_TEAMMATES=0

OFFENSE_AGENT_FILE=$BASE_DIR/matias_hfo/agents/q_agent/learning_agent.py
OFFENSE_AGENT_SAVE="q_agent_4000ep_1dumbdefense"
DEFENSE_AGENT_FILE=$BASE_DIR/matias_hfo/agents/goalkeeper/test_agent.py

echo $HFO
$HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
 --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
 --offense-on-ball 11  --trials $NUM_EPISODES --deterministic --fullstate \
 --no-logging \
 --headless &
# --no-sync  &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 5
echo "Connect to player"
$PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
--num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_EPISODES \
--save_file=$OFFENSE_AGENT_SAVE &
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

