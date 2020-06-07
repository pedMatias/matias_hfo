#!/bin/bash
killall -9 rcssserver

export PYTHONPATH=/home/matias/Desktop/HFO:$PYTHONPATH
export PYTHONPATH=/home/matias/Desktop/HFO/matias_hfo:$PYTHONPATH
echo $PYTHONPATH

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python

# Train config:
NUM_TRAIN_EP=500
NUM_TEST_EP=50
NUM_REPETITIONS=20
NUM_EPISODES=$(($(($(($NUM_TRAIN_EP + $NUM_TEST_EP)) * $NUM_REPETITIONS)) + $NUM_TEST_EP))
echo "Episodes: $NUM_EPISODES"

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

OFFENSE_AGENT_FILE=$BASE_DIR/matias_hfo/agents/q_agent_1teammate_v1/train_player_w_static.py
# DEFENSE_AGENT_FILE=$BASE_DIR/matias_hfo/agents/goalkeeper/player_agent.py
DEFENSE_AGENT_FILE=$BASE_DIR/matias_hfo/agents/goalkeeper/goalkeeper_v2.py
STATIC_AGENT_FILE=$BASE_DIR/matias_hfo/agents/fixed_teammate/static_agent.py

$HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
 --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
 --offense-on-ball 6  --trials $NUM_EPISODES --deterministic --fullstate \
 --no-logging --frames-per-trial 500 --untouched-time 400 \
 --headless >> hfo.log &
# --no-sync >> hfo.log &
# Sleep is needed to make sure doesn't get connected too soon

sleep 2
echo "Connect to Main player"
$PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
--num_teammates=$TOTAL_TEAMMATES --num_train_ep=$NUM_TRAIN_EP \
--num_test_ep=$NUM_TEST_EP --num_repetitions=$NUM_REPETITIONS
echo "PLayer connected"

sleep 2
echo "Connect to Static player"
$PYTHON $STATIC_AGENT_FILE  --num_episodes=$NUM_EPISODES \
--num_opponents=$TOTAL_OPPONENTS --num_teammates=$TOTAL_TEAMMATES
echo "PLayer connected"

sleep 2
echo "Connect Defense Player"
$PYTHON $DEFENSE_AGENT_FILE  --num_episodes=$NUM_EPISODES \
--num_offenses=$TOTAL_OFFENSES --num_defenses=$(($TOTAL_DEFENSES-1)) &

# .py &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

