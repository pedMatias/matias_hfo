#!/bin/bash
killall -9 rcssserver

PORT=6000

# Directories:
BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
MODELS_DIR=$BASE_DIR/matias_hfo/models
AGENTS_DIR=$BASE_DIR/matias_hfo/multi_agents

# Scripts Files:
PLAYER_FILE=$AGENTS_DIR/dqn_agent/train/base_model/run_player.py
TRAIN_FILE=$AGENTS_DIR/dqn_agent/train/train_model.py
CREATE_MODEL_FILE=$AGENTS_DIR/dqn_agent/train/base_model/create_model.py

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

GAME_SET_UP="${TOTAL_OFFENSES}vs${TOTAL_DEFENSES}"
DIR_NAME="${MODELS_DIR}/${GAME_SET_UP}/base"


# ***********************  SET UP Train  ***********************************
DIR_NAME="${MODELS_DIR}/${GAME_SET_UP}"
mkdir $DIR_NAME
DIR_NAME="${DIR_NAME}/base"
mkdir $DIR_NAME

# Create Base Model:
BASE_MODEL="${DIR_NAME}/base.model"
$PYTHON $CREATE_MODEL_FILE --num_opponents=$TOTAL_OPPONENTS \
--num_teammates=$TOTAL_TEAMMATES --dir=$DIR_NAME --model_name=$BASE_MODEL &


## ********************  1. TRAIN SOLO  *****************************************
#TEAMMATE_NAME="yushan"
#NUM_TEST_EPISODES=100
#NUM_EPISODES_LIST=(1000 5000 5000)  # 100k espisodes
#EPSILONS_LIST=(1.0 0.5 0.2)
#LEARNING_BOOST="true"
#PASS_BALL="false"
#for i in {0..2}
#do
#  STEP=$i
#  NUM_EPISODES=${NUM_EPISODES_LIST[i]}
#  EPSILON=${EPSILONS_LIST[i]}
#
#  # ******************* STAGE 1: Explore for the first time *******************
#  MODE="exploration"
#  echo "[${MODE}] STEP ${STEP} "
#  $HFO --offense-team $TEAMMATE_NAME --offense-agents $NUM_OFFENSES \
#  --offense-npcs $NUM_OFFENSES_NPCS --defense-agents $NUM_DEFENSES \
#  --defense-npcs $NUM_DEFENSES_NPCS --offense-on-ball $((-1))  \
#  --trials $NUM_EPISODES --deterministic --fullstate --no-logging \
#  --frames-per-trial 500 --untouched-time 300 --port $PORT --seed $SEED \
#  --headless &
#  # --no-sync &
#
#  MODEL_FILE="${DIR_NAME}/$((${STEP}-1)).model"
#  if [ $STEP = 0 ]
#  then
#    MODEL_FILE=$BASE_MODEL
#  fi
#  sleep 2
#  echo "MODEL: ${MODEL_FILE}"
#  $PYTHON $PLAYER_FILE --mode=$MODE  --num_teammates=$TOTAL_TEAMMATES \
#  --num_opponents=$TOTAL_OPPONENTS --num_episodes=$NUM_EPISODES \
#  --step=$STEP --dir=$DIR_NAME --port=$PORT --team_name=$TEAMMATE_NAME \
#  --model_file=$MODEL_FILE --epsilon=$EPSILON --pass_ball=$PASS_BALL\
#  --learning_boost=$LEARNING_BOOST &
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#  # ******************* STAGE 2: Train offline *******************
#  echo "TRAIN ${STEP}: ${NUM_EPISODES} episodes; ${EPSILON} epsilon;"
#  $PYTHON $TRAIN_FILE --num_opponents=$TOTAL_OPPONENTS \
#    --num_teammates=$TOTAL_TEAMMATES --dir=$DIR_NAME --step=$STEP \
#    --team_name=$TEAM_NAME &
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#  # ******************* STAGE 3: Test *******************
#  MODE="testing"
#  echo "[${MODE}] STEP ${STEP}"
#  MODEL_FILE="${DIR_NAME}/${STEP}.model"
#
#  echo "[TEST: ${TEST_IDX}/${NUM_SUB_MODELS}] ${NUM_TEST_EPISODES} episodes"
#  $HFO --offense-team $TEAMMATE_NAME --offense-agents $NUM_OFFENSES \
#  --offense-npcs $NUM_OFFENSES_NPCS --defense-agents $NUM_DEFENSES \
#  --defense-npcs $NUM_DEFENSES_NPCS --offense-on-ball $((-1))  \
#  --trials $NUM_TEST_EPISODES --deterministic --fullstate --no-logging \
#  --frames-per-trial 500 --untouched-time 300 --port $PORT --seed $SEED \
#  --headless &
#  # --no-sync &
#
#  sleep 3
#  echo "Model: ${MODEL_FILE}"
#  $PYTHON $PLAYER_FILE --mode=$MODE --num_opponents=$TOTAL_OPPONENTS \
#  --num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_TEST_EPISODES \
#  --step=$STEP --dir=$DIR_NAME --port=$PORT \
#  --team_name=$TEAMMATE_NAME --model_file=$MODEL_FILE --epsilon=0  &
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#done


# ******************** 2.TRAIN W TEAMMATES (FIXED) ****************************
BASE_MODEL="${DIR_NAME}/base.model"
TEAMMATE_NAME="base"
SEED=123
NUM_TEST_EPISODES=100
NUM_EPISODES_LIST=(5000 5000 5000 5000 5000 5000 5000)  # 25k espisodes
EPSILONS_LIST=(0.7 0.35 0.5 0.25 0.3 0.15 0.05)
LEARNING_BOOST="true"
PASS_BALL="true"
for i in {4..6}
do
  STEP=$i
  NUM_EPISODES=${NUM_EPISODES_LIST[i]}
  EPSILON=${EPSILONS_LIST[i]}

#  # ******************* STAGE 1: Explore for the first time *******************
#  MODE="exploration"
#  echo "[${MODE}] STEP ${STEP} "
#  $HFO --offense-team $TEAMMATE_NAME --offense-agents $NUM_OFFENSES \
#  --offense-npcs $NUM_OFFENSES_NPCS --defense-agents $NUM_DEFENSES \
#  --defense-npcs $NUM_DEFENSES_NPCS --offense-on-ball $((-1))  \
#  --trials $NUM_EPISODES --deterministic --fullstate --no-logging \
#  --frames-per-trial 500 --untouched-time 300 --port $PORT --seed $SEED \
#  --headless &
#  # --no-sync &
#
#  MODEL_FILE="${DIR_NAME}/$((${STEP}-1)).model"
#  if [ $STEP = 0 ]
#  then
#    MODEL_FILE=$BASE_MODEL
#  fi
#  sleep 2
#  echo "MODEL: ${MODEL_FILE}"
#  $PYTHON $PLAYER_FILE --mode=$MODE  --num_teammates=$TOTAL_TEAMMATES \
#  --num_opponents=$TOTAL_OPPONENTS --num_episodes=$NUM_EPISODES \
#  --step=$STEP --dir=$DIR_NAME --port=$PORT --team_name=$TEAMMATE_NAME \
#  --model_file=$MODEL_FILE --epsilon=$EPSILON --pass_ball=$PASS_BALL \
#  --learning_boost=$LEARNING_BOOST &
#
#  trap "kill -TERM -$$" SIGINT
#  wait

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

  echo "[TEST: ${TEST_IDX}/${NUM_SUB_MODELS}] ${NUM_TEST_EPISODES} episodes"
  $HFO --offense-team $TEAMMATE_NAME --offense-agents $NUM_OFFENSES \
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
  --step=$STEP --dir=$DIR_NAME --port=$PORT \
  --team_name=$TEAMMATE_NAME --model_file=$MODEL_FILE --epsilon=0  &

  trap "kill -TERM -$$" SIGINT
  wait


  echo ""
  echo "----------------------- STEP $(STEP) ENDED -------------------------"
  echo ""
  sleep 3

done


# ******************** 2. TRAIN W TEAMMATES **************************************
BASE_MODEL="${DIR_NAME}/backup_6.model"
TEAMMATE_NAME="base"
NUM_TEST_EPISODES=500
NUM_EPISODES_LIST=(0 0 0 0 0 0 5000 5000 5000 5000)  # 25k espisodes
EPSILONS_LIST=(1. 1. 1. 1. 1. 1. 0.3 0.2 0.1 0.05)
LEARNING_BOOST="false"
PASS_BALL="false"
for i in {10..10}
do
  STEP=$i
  NUM_EPISODES=5000  # ${NUM_EPISODES_LIST[i]}
  EPSILON=0.05  # ${EPSILONS_LIST[i]}

#  # ******************* STAGE 1: Explore for the first time *******************
#  MODE="exploration"
#  echo "[${MODE}] STEP ${STEP} "
#  $HFO --offense-team $TEAMMATE_NAME --offense-agents $NUM_OFFENSES \
#  --offense-npcs $NUM_OFFENSES_NPCS --defense-agents $NUM_DEFENSES \
#  --defense-npcs $NUM_DEFENSES_NPCS --offense-on-ball $((-1))  \
#  --trials $NUM_EPISODES --deterministic --fullstate --no-logging \
#  --frames-per-trial 500 --untouched-time 300 --port $PORT --seed $SEED \
#  --headless &
#  # --no-sync &
#
#  MODEL_FILE="${DIR_NAME}/$((${STEP}-1)).model"
#  if [ $STEP = 0 ]
#  then
#    MODEL_FILE=$BASE_MODEL
#  fi
#  sleep 2
#  echo "MODEL: ${MODEL_FILE}"
#  $PYTHON $PLAYER_FILE --mode=$MODE  --num_teammates=$TOTAL_TEAMMATES \
#  --num_opponents=$TOTAL_OPPONENTS --num_episodes=$NUM_EPISODES \
#  --step=$STEP --dir=$DIR_NAME --port=$PORT --team_name=$TEAMMATE_NAME \
#  --model_file=$MODEL_FILE --epsilon=$EPSILON --pass_ball=$PASS_BALL \
#  --learning_boost=$LEARNING_BOOST &  # 2> /dev/null &
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#  # ******************* STAGE 2: Train offline *******************
#  echo "TRAIN ${STEP}: ${NUM_EPISODES} episodes; ${EPSILON} epsilon;"
#  $PYTHON $TRAIN_FILE --num_opponents=$TOTAL_OPPONENTS \
#    --num_teammates=$TOTAL_TEAMMATES --dir=$DIR_NAME --step=$STEP \
#    --team_name=$TEAM_NAME &
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#    # ******************* STAGE 3: Test *******************
#  MODE="testing"
#  echo "[${MODE}] STEP ${STEP}"
#  MODEL_FILE="${DIR_NAME}/${STEP}.model"
#
#  echo "[TEST: ${TEST_IDX}/${NUM_SUB_MODELS}] ${NUM_TEST_EPISODES} episodes"
#  $HFO --offense-team $TEAMMATE_NAME --offense-agents $NUM_OFFENSES \
#  --offense-npcs $NUM_OFFENSES_NPCS --defense-agents $NUM_DEFENSES \
#  --defense-npcs $NUM_DEFENSES_NPCS --offense-on-ball $((-1))  \
#  --trials $NUM_TEST_EPISODES --deterministic --fullstate --no-logging \
#  --frames-per-trial 500 --untouched-time 300 --port $PORT --seed $SEED \
#  --headless &
#  # --no-sync &
#
#  sleep 3
#  echo "Model: ${MODEL_FILE}"
#  $PYTHON $PLAYER_FILE --mode=$MODE --num_opponents=$TOTAL_OPPONENTS \
#  --num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_TEST_EPISODES \
#  --step=$STEP --dir=$DIR_NAME --port=$PORT \
#  --team_name=$TEAMMATE_NAME --model_file=$MODEL_FILE --epsilon=0  &
#
#  trap "kill -TERM -$$" SIGINT
#  wait

  # ******************* STAGE 3: Test *******************
  MODE="testing"
  echo "[${MODE}] STEP ${STEP}"
  MODEL_BASE="${DIR_NAME}/${STEP}.model"
  NUM_SUB_MODELS=$(( $(ls $MODEL_BASE.* | wc -l) - 1 ))

  for TEST_IDX in $( seq 0 $NUM_SUB_MODELS)
    do

    echo "[TEST: ${TEST_IDX}/${NUM_SUB_MODELS}] ${NUM_TEST_EPISODES} episodes"
    $HFO --offense-team $TEAMMATE_NAME --offense-agents $NUM_OFFENSES \
    --offense-npcs $NUM_OFFENSES_NPCS --defense-agents $NUM_DEFENSES \
    --defense-npcs $NUM_DEFENSES_NPCS --offense-on-ball $((-1))  \
    --trials $NUM_TEST_EPISODES --deterministic --fullstate --no-logging \
    --frames-per-trial 500 --untouched-time 300 --port $PORT --seed $SEED \
    --headless &
    # --no-sync &

    sleep 3
    MODEL_FILE="${MODEL_BASE}.${TEST_IDX}"
    echo "Model: ${MODEL_FILE}"
    $PYTHON $PLAYER_FILE --mode=$MODE --num_opponents=$TOTAL_OPPONENTS \
    --num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_TEST_EPISODES \
    --step=$STEP --test_id=$TEST_IDX --dir=$DIR_NAME --port=$PORT \
    --team_name=$TEAMMATE_NAME --model_file=$MODEL_FILE --epsilon=0 \
    2> /dev/null &

    trap "kill -TERM -$$" SIGINT
    wait
  done

  echo ""
  echo "----------------------- STEP $(STEP) ENDED -------------------------"
  echo ""
  sleep 3

done