#!/bin/bash
killall -9 rcssserver

export PYTHONPATH=/home/matias/Desktop/HFO:$PYTHONPATH
export PYTHONPATH=/home/matias/Desktop/HFO/matias_hfo:$PYTHONPATH
echo $PYTHONPATH

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
MODULE_DIR=$BASE_DIR/matias_hfo/agents
DATA_DIR=$BASE_DIR/matias_hfo/data

NUM_EPISODES=6

NUM_DEFENSES=2
NUM_DEFENSES_NPCS=0
TOTAL_DEFENSES=$(($NUM_DEFENSES + $NUM_DEFENSES_NPCS))
TOTAL_OPPONENTS=$TOTAL_DEFENSES
echo "TOTAL_OPPONENTS: $TOTAL_OPPONENTS"

NUM_OFFENSES=0
NUM_OFFENSES_NPCS=1
TOTAL_OFFENSES=$(($NUM_OFFENSES + $NUM_OFFENSES_NPCS))
TOTAL_TEAMMATES=$(($TOTAL_OFFENSES - 1))
echo "TOTAL_TEAMMATES: $TOTAL_TEAMMATES"

GOALIE_AGENT_FILE=$MODULE_DIR/fixed_agents/goalkeeper/player_agent.py
DEFENSE_AGENT_FILE=$MODULE_DIR/fixed_agents/defense/hand_coded_defense_agent.py
STATIC_AGENT_FILE=$MODULE_DIR/fixed_agents/fixed_teammate/static_agent.py

$HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
 --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
 --offense-on-ball 6  --trials $NUM_EPISODES --deterministic --fullstate \
 --frames-per-trial 300 --no-logging \
 --no-sync  &
# --headless &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 2
echo "Connect Defense Player"
$PYTHON $GOALIE_AGENT_FILE  --num_episodes=$NUM_EPISODES \
--num_offenses=$TOTAL_OFFENSES --num_defenses=$(($TOTAL_DEFENSES-1)) &

sleep 2
echo "Connect to Main player"
$PYTHON $DEFENSE_AGENT_FILE  &

# .py &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

