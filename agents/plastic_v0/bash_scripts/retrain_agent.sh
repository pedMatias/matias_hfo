#!/bin/bash
killall -9 rcssserver

export PYTHONPATH=/home/matias/Desktop/HFO:$PYTHONPATH
export PYTHONPATH=/home/matias/Desktop/HFO/matias_hfo:$PYTHONPATH
echo $PYTHONPATH

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
MODULE_DIR=$BASE_DIR/matias_hfo/agents

# Train config:
NUM_TRAIN_EP=2500
NUM_TEST_EP=18
NUM_REPETITIONS=20
NUM_EPISODES=$(($(($NUM_TRAIN_EP + $NUM_TEST_EP)) * $NUM_REPETITIONS))
echo "Episodes: $NUM_EPISODES"

NUM_DEFENSES=2
NUM_DEFENSES_NPCS=0
TOTAL_DEFENSES=$(($NUM_DEFENSES + $NUM_DEFENSES_NPCS))
TOTAL_OPPONENTS=$TOTAL_DEFENSES
echo "TOTAL_OPPONENTS: $TOTAL_OPPONENTS"

NUM_OFFENSES=1
NUM_OFFENSES_NPCS=1
TOTAL_OFFENSES=$(($NUM_OFFENSES + $NUM_OFFENSES_NPCS))
TOTAL_TEAMMATES=$(($TOTAL_OFFENSES - 1))
echo "TOTAL_TEAMMATES: $TOTAL_TEAMMATES"

# GOALIE_AGENT_FILE=$MODULE_DIR/fixed_agents/goalkeeper/good_teammate.py
GOALIE_AGENT_FILE=$MODULE_DIR/fixed_agents/goalkeeper/goalkeeper_v2.py
DEFENSE_AGENT_FILE=$MODULE_DIR/fixed_agents/defense/hand_coded_defense_agent.py
STATIC_AGENT_FILE=$MODULE_DIR/fixed_agents/fixed_teammate/player_agent.py

OFFENSE_AGENT_FILE=$MODULE_DIR/plastic_v0/dqn_train.py
MODEL=/home/matias/Desktop/HFO/matias_hfo/data/new_dqn_50600ep_1op_2020-06-11_18:44:00/agent_model

$HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
 --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
 --offense-on-ball 6  --trials $NUM_EPISODES --deterministic --fullstate \
 --no-logging --frames-per-trial 500 --untouched-time 100  --message-size 10 \
 --headless >> hfo.log &
# --no-sync >> hfo.log &

sleep 2
echo "Connect to Main player"
$PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
--num_teammates=$TOTAL_TEAMMATES --num_train_ep=$NUM_TRAIN_EP \
--num_test_ep=$NUM_TEST_EP --num_repetitions=$NUM_REPETITIONS \
--load_file=$MODEL_FILE --retrain=$(true)  &

# sleep 3
# echo "Connect to Fixed player"
# $PYTHON $STATIC_AGENT_FILE  --num_episodes=$NUM_EPISODES \
# --num_opponents=$TOTAL_OPPONENTS --num_teammates=$TOTAL_TEAMMATES \
# --wait_for_teammate=true &

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

