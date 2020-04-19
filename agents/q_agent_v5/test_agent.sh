#!/bin/bash
killall -9 rcssserver

export PYTHONPATH=/home/matias/Desktop/HFO:$PYTHONPATH
export PYTHONPATH=/home/matias/Desktop/HFO/matias_hfo:$PYTHONPATH
echo $PYTHONPATH

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python

# Train config:
OFFENSE_AGENT_FILE=$BASE_DIR/matias_hfo/agents/q_agent_v5/test_agent.py
SAVE_DIR=$BASE_DIR/matias_hfo/data/q_agent_train_2020-04-18_20:06:00
Q_TABLE=$SAVE_DIR/q_table.npy

NUM_EPISODES=10

NUM_DEFENSES=1
NUM_DEFENSES_NPCS=0
TOTAL_OPPONENTS=$(($NUM_DEFENSES + $NUM_DEFENSES_NPCS))
echo "TOTAL_OPPONENTS: $TOTAL_OPPONENTS"

NUM_OFFENSES=1
NUM_OFFENSES_NPCS=0
TOTAL_TEAMMATES=$((NUM_OFFENSES + NUM_OFFENSES_NPCS - 1))
echo "TOTAL_TEAMMATES: $TOTAL_TEAMMATES"

DEFENSE_AGENT_FILE=$BASE_DIR/matias_hfo/agents/goalkeeper/player_agent.py

$HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
 --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
 --offense-on-ball 11  --trials $NUM_EPISODES --deterministic --fullstate \
 --frames-per-trial 200 --no-logging --no-sync  &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 5
echo "Connect to player"
$PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
--num_teammates=$TOTAL_TEAMMATES --num_ep=$NUM_EPISODES --load_file=$Q_TABLE \
--save_dir=$SAVE_DIR &
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

