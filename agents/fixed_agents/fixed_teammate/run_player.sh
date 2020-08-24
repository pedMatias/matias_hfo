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
NUM_DEFENSES_NPCS=2
TOTAL_DEFENSES=$(($NUM_DEFENSES + $NUM_DEFENSES_NPCS))
TOTAL_OPPONENTS=$TOTAL_DEFENSES
echo "TOTAL_OPPONENTS: $TOTAL_OPPONENTS"

NUM_OFFENSES=2
NUM_OFFENSES_NPCS=0
TOTAL_OFFENSES=$(($NUM_OFFENSES + $NUM_OFFENSES_NPCS))
TOTAL_TEAMMATES=$(($TOTAL_OFFENSES - 1))
echo "TOTAL_TEAMMATES: $TOTAL_TEAMMATES"

# Players files:
AGENTS_DIR=$PROJECT_DIR/agents/fixed_agents
FIXED_AGENT_FILE=$AGENTS_DIR/fixed_teammate/helios_teammate.py
STATIC_AGENT_FILE=$AGENTS_DIR/fixed_teammate/static_teammate.py

DEFENSE_AGENT_FILE=$AGENTS_DIR/goalkeeper/good_goalkeeper.py

echo $HFO
$HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
 --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
  --trials $NUM_EPISODES \
 --deterministic --fullstate --no-sync --no-logging &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 2
echo "Connect to Fixed player"
$PYTHON $FIXED_AGENT_FILE  --num_episodes=$NUM_EPISODES \
--num_opponents=$TOTAL_OPPONENTS --num_teammates=$TOTAL_TEAMMATES \
--wait_for_teammate=$false &

sleep 2
echo "Connect to Fixed player"
$PYTHON $FIXED_AGENT_FILE  --num_episodes=$NUM_EPISODES \
--num_opponents=$TOTAL_OPPONENTS --num_teammates=$TOTAL_TEAMMATES \
--wait_for_teammate=$false &

#sleep 2
#echo "Connect Defense Player"
#$PYTHON $DEFENSE_AGENT_FILE  --num_episodes=$NUM_EPISODES \
#--num_offenses=$TOTAL_OFFENSES --num_defenses=$(($TOTAL_DEFENSES-1)) &

# .py &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

