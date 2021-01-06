#!/bin/bash

# Directories:
BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
MODELS_DIR=$BASE_DIR/matias_hfo/models
AGENTS_DIR=$BASE_DIR/matias_hfo/multi_agents

# Scripts Files:
PLAYER_FILE=$AGENTS_DIR/dqn_agent/run/run_player.py
TRAIN_FILE=$AGENTS_DIR/dqn_agent/train/train_model.py

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

# killall -9 rcssserver
# "gliders" "helios" "aut" "axiom" "cyrus"
PORT=6200
SEED=123
GAME_SET_UP="${TOTAL_OFFENSES}vs${TOTAL_DEFENSES}"
TEAM_NAME="gliders"
DIR_NAME="${MODELS_DIR}/${GAME_SET_UP}/${TEAM_NAME}"


# *********************************  TRAIN ***********************************
BASE_MODEL="${DIR_NAME}/base.model"
NUM_TEST_EPISODES=500
NUM_EPISODES_LIST=(25000 25000 25000 25000)
EPSILONS_LIST=(0.3 0.2 0.1 0.05)
for i in {0..3}
do
  STEP=$i
  NUM_EPISODES=${NUM_EPISODES_LIST[i]}
  EPSILON=${EPSILONS_LIST[i]}

  # ******************* STAGE 1: Explore for the first time *******************
  MODE="exploration"
  echo "[${MODE}] STEP ${STEP} "
  $HFO --offense-team $TEAM_NAME --offense-agents $NUM_OFFENSES \
  --offense-npcs $NUM_OFFENSES_NPCS --defense-agents $NUM_DEFENSES \
  --defense-npcs $NUM_DEFENSES_NPCS --offense-on-ball $((-1))  \
  --trials $NUM_EPISODES --deterministic --fullstate --no-logging \
  --frames-per-trial 500 --untouched-time 300 --port $PORT --seed $SEED \
  --headless &
  # --no-sync &

  MODEL_FILE="${DIR_NAME}/$((${STEP}-1)).model"
  if [ $STEP = 0 ]
  then
    MODEL_FILE=$BASE_MODEL
  fi
  sleep 2
  echo "MODEL: ${MODEL_FILE}"
  $PYTHON $PLAYER_FILE --mode=$MODE  --num_teammates=$TOTAL_TEAMMATES \
  --num_opponents=$TOTAL_OPPONENTS --num_episodes=$NUM_EPISODES \
  --step=$STEP --dir=$DIR_NAME --port=$PORT --team_name=$TEAM_NAME \
  --model_file=$MODEL_FILE --epsilon=$EPSILON 2> /dev/null &

  trap "kill -TERM -$$" SIGINT
  wait

  # ******************* STAGE 2: Train offline *******************
  echo "TRAIN ${STEP}: ${NUM_EPISODES} episodes; ${EPSILON} epsilon;"
  $PYTHON $TRAIN_FILE --num_opponents=$TOTAL_OPPONENTS \
    --num_teammates=$TOTAL_TEAMMATES --dir=$DIR_NAME --step=$STEP \
    --team_name=$TEAM_NAME &

  trap "kill -TERM -$$" SIGINT
  wait

  # ******************* STAGE 3: Test *******************
  MODE="testing"
  echo "[${MODE}] STEP ${STEP}"
  MODEL_FILE="${DIR_NAME}/${STEP}.model"
  TEST_IDX=0

  echo "[TEST: ${TEST_IDX}/${NUM_SUB_MODELS}] ${NUM_TEST_EPISODES} episodes"
  $HFO --offense-team $TEAM_NAME --offense-agents $NUM_OFFENSES \
  --offense-npcs $NUM_OFFENSES_NPCS --defense-agents $NUM_DEFENSES \
  --defense-npcs $NUM_DEFENSES_NPCS --offense-on-ball $((-1))  \
  --trials $NUM_TEST_EPISODES --deterministic --fullstate --no-logging \
  --frames-per-trial 500 --untouched-time 300 --port $PORT --seed $SEED \
  --headless &
  # --no-sync &

  sleep 3
  echo "Model: ${MODEL_FILE}"
  $PYTHON $PLAYER_FILE --mode=$MODE --num_opponents=$TOTAL_OPPONENTS \
  --num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_TEST_EPISODES \
  --step=$STEP --test_id=$TEST_IDX --dir=$DIR_NAME --port=$PORT \
  --team_name=$TEAM_NAME --model_file=$MODEL_FILE --epsilon=0 \
  &  # 2> /dev/null &

  trap "kill -TERM -$$" SIGINT
  wait

  echo ""
  echo "----------------------- STEP $(STEP) ENDED -------------------------"
  echo ""
  sleep 3

done
