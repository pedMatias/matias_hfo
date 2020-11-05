#!/bin/bash
killall -9 rcssserver

PORT=6000

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
MODELS_DIR=$BASE_DIR/matias_hfo/models
AGENTS_DIR=$BASE_DIR/matias_hfo/plastic_agent

echo "HFO: ${HFO}"
echo "PYTHON: ${PYTHON}"

export PYTHONPATH=$BASE_DIR:$PYTHONPATH
export PYTHONPATH=$BASE_DIR/matias_hfo:$PYTHONPATH

NUM_DEFENSES=0
NUM_DEFENSES_NPCS=2
TOTAL_DEFENSES=$(($NUM_DEFENSES + $NUM_DEFENSES_NPCS))
TOTAL_OPPONENTS=$TOTAL_DEFENSES
echo "TOTAL_OPPONENTS: $TOTAL_OPPONENTS"

NUM_OFFENSES=1
NUM_OFFENSES_NPCS=1
TOTAL_OFFENSES=$(($NUM_OFFENSES + $NUM_OFFENSES_NPCS))
TOTAL_TEAMMATES=$(($TOTAL_OFFENSES - 1))
echo "TOTAL_TEAMMATES: $TOTAL_TEAMMATES"

STEP=0
EPSILON=0
NO_SAVE="true"
NUM_TEST_EPISODES=10

BASE_MODEL=$MODELS_DIR/base/agent_model

MODE="testing"
TEAMMATE_NAME="yushan"
MODEL_FILE="${MODELS_DIR}/${TEAMMATE_NAME}/0.model.2"
# MODEL_FILE=${BASE_MODEL}
PLAYER_FILE=$AGENTS_DIR/run_player.py

echo "[${MODE}] STEP ${STEP}"

echo "[TEST: ${TEST_IDX}/${NUM_SUB_MODELS}] ${NUM_TEST_EPISODES} episodes"
$HFO --offense-team $TEAMMATE_NAME --offense-agents $NUM_OFFENSES \
--offense-npcs $NUM_OFFENSES_NPCS --defense-agents $NUM_DEFENSES \
--defense-npcs $NUM_DEFENSES_NPCS --offense-on-ball $((-1))  \
--trials $NUM_TEST_EPISODES --deterministic --fullstate --no-logging \
--frames-per-trial 500 --untouched-time 300 --port $PORT \
 --no-sync &
#&
# --headless &

sleep 2
echo "Model: ${MODEL_FILE}"
$PYTHON $PLAYER_FILE --mode=$MODE --num_opponents=$TOTAL_OPPONENTS \
--num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_TEST_EPISODES \
--step=$STEP --dir=$DIR_NAME --port=$PORT \
--no_save=$NO_SAVE --team_name=$TEAMMATE_NAME --model_file=$MODEL_FILE \
--epsilon=0 &

trap "kill -TERM -$$" SIGINT
wait

echo ""
echo "----------------------- STEP $(STEP) ENDED -------------------------"
echo ""
