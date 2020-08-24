#!/bin/bash
killall -9 rcssserver

export PYTHONPATH=/home/matias/Desktop/HFO:$PYTHONPATH
export PYTHONPATH=/home/matias/Desktop/HFO/matias_hfo:$PYTHONPATH
echo $PYTHONPATH

PORT=6010

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
GOALKEEPER_DIR=$BASE_DIR/matias_hfo/agents/fixed_agents/goalkeeper
TEAMMATES_DIR=$BASE_DIR/matias_hfo/agents/fixed_agents/fixed_teammate
AGENTS_DIR=$BASE_DIR/matias_hfo/agents/dqn_v1

# Agents Files:
# DEFENSE_AGENT_FILE=$GOALKEEPER_DIR/good_teammate.py
DEFENSE_AGENT_FILE=$GOALKEEPER_DIR/goalkeeper_v2.py
STATIC_AGENT_FILE=$TEAMMATES_DIR/static_agent.py
OFFENSE_AGENT_FILE=$AGENTS_DIR/dumb/test_agent.py

# Train config:
NUM_EPISODES=6

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

$HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
 --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
 --offense-on-ball 7  --trials $NUM_EPISODES --deterministic --fullstate \
 --no-logging --frames-per-trial 300 --untouched-time 60 --port $PORT\
 --no-sync >> hfo.log &
# --headless >> hfo.log &

sleep 2
echo "Connect to Main player"
$PYTHON $OFFENSE_AGENT_FILE  --num_games=$NUM_EPISODES --port=$PORT &

sleep 2
echo "Connect to Static player"
$PYTHON $STATIC_AGENT_FILE  --num_episodes=$NUM_EPISODES --port=$PORT \
--num_opponents=$TOTAL_OPPONENTS --num_teammates=$TOTAL_TEAMMATES &

sleep 2
echo "Connect Defense Player"
$PYTHON $DEFENSE_AGENT_FILE  --num_episodes=$NUM_EPISODES --port=$PORT \
--num_offenses=$TOTAL_OFFENSES --num_defenses=$(($TOTAL_DEFENSES-1)) &

echo "GAME READY!"
# .py &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

