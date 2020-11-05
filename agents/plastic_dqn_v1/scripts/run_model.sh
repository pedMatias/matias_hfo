#!/bin/bash
killall -9 rcssserver

PORT=6000

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
AGENTS_DIR=$BASE_DIR/matias_hfo/agents
MODELS_DIR=$BASE_DIR/matias_hfo/models

export PYTHONPATH=$BASE_DIR:$PYTHONPATH
export PYTHONPATH=$BASE_DIR/matias_hfo:$PYTHONPATH

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

DIR_NAME=$MODELS_DIR/custom_hfo_v0
AGENT_MODEL_NAME=$DIR_NAME/custom_hfo_2.model

BASE_MODEL_DQN=$MODELS_DIR/base/agent_model
TEAMMATE_FILE=$AGENTS_DIR/fixed_agents/fixed_teammate/hfo_custom_agent.py


# ** TEST:
echo "[TEST: {$AGENT_MODEL_NAME}] With teammate {$TEAMMATE_FILE}"

$HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
 --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
 --offense-on-ball $((-1))  --trials $NUM_TEST_EPISODES --deterministic \
 --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
 --port $PORT \
 --no-sync >> hfo.log &

sleep 3
OFFENSE_AGENT_FILE=$AGENTS_DIR/plastic_dqn_v1/test_player.py
$PYTHON $OFFENSE_AGENT_FILE --num_opponents=$TOTAL_OPPONENTS \
--num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_TEST_EPISODES \
--team_name=$TEAM_NAME --new_model=$NEW_MODEL \
--starts_fixed_position=$STARTS_FIXED_POSITION --step=$STEP \
--model_idx=$TEST_IDX --dir=$DIR_NAME --port=$PORT &

sleep 3
$PYTHON $TEAMMATE_FILE  --epsilon=0  &

trap "kill -TERM -$$" SIGINT
wait