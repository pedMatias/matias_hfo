#!/bin/bash
killall -9 rcssserver

export PYTHONPATH=/home/matias/Desktop/HFO:$PYTHONPATH
export PYTHONPATH=/home/matias/Desktop/HFO/matias_hfo:$PYTHONPATH
echo $PYTHONPATH

PORT=6000

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
AGENTS_DIR=$BASE_DIR/matias_hfo/agents
DATA_DIR=$BASE_DIR/matias_hfo/data

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

#DIR_NAME=$DATA_DIR/complete_train_complete_actions$(date +"%Y-%m-%dT%T")
DIR_NAME=$DATA_DIR/helios_v0
mkdir $DIR_NAME
AGENT_FILE=$AGENTS_DIR/plastic_dqn_v1/explore_player.py

# ************************************* STAGE 1 ********************************
# Train with a helios teammates, against two helios defenses
STARTS_FIXED_POSITION="false"
TEAM_NAME="helios"

NUM_TEST_EPISODES=1000
NUM_EPISODES_LIST=(25000 25000 25000 25000)  # 100k espisodes
EPSILONS_LIST=(0.6 0.4 0.2 0.1)

for i in {3..3}
do
  STEP=$i
  NUM_EPISODES=${NUM_EPISODES_LIST[i]}
  EPSILON=${EPSILONS_LIST[i]}

#  #STAGE 1: Explore for the first time
#  echo "EXPLORATION ${STEP}: ${NUM_EPISODES} episodes; ${EPSILON} epsilon;"
#  $HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
#  --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
#  --offense-on-ball $((-1))  --trials $NUM_EPISODES --deterministic \
#  --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
#  --port $PORT \
#  --headless >> hfo.log &
#  #--no-sync >> hfo.log &
#
#  sleep 2
#  $PYTHON $AGENT_FILE  --num_teammates=$TOTAL_TEAMMATES \
#  --num_opponents=$TOTAL_OPPONENTS --num_episodes=$NUM_EPISODES \
#  --team_name=$TEAM_NAME --starts_fixed_position=$STARTS_FIXED_POSITION \
#  --step=$STEP --epsilon=$EPSILON --dir=$DIR_NAME --port=$PORT &
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#
#  # ** STAGE 2: TRAIN OFFLINE:
#  TRAIN_SCRIPT=$AGENTS_DIR/plastic_dqn_v1/train_model.py
#  echo "TRAIN ${STEP}: ${NUM_EPISODES} episodes; ${EPSILON} epsilon;"
#  $PYTHON $TRAIN_SCRIPT --num_opponents=$TOTAL_OPPONENTS \
#    --num_teammates=$TOTAL_TEAMMATES --dir=$DIR_NAME --step=$STEP \
#    --team_name=$TEAM_NAME &
#
#  trap "kill -TERM -$$" SIGINT
#  wait

  # ** STAGE 3.1: TEST New models:
  echo "[TEST: STEP $STEP] New Models"
  NEW_MODEL="true"
  MODEL_BASE="new_${TEAM_NAME}_${STEP}.model"
  NUM_SUB_MODELS=$(ls $DIR_NAME/$MODEL_BASE.* | wc -l)

  for TEST_IDX in $( seq 1 $NUM_SUB_MODELS)
    do

    echo "[TEST: New ${TEST_IDX}/${NUM_SUB_MODELS}] ${NUM_TEST_EPISODES} episodes"

    MODEL_FILE="${MODEL_BASE}.${test_idx}"
    $HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
     --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
     --offense-on-ball $((-1))  --trials $NUM_TEST_EPISODES --deterministic \
     --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
     --port $PORT \
     --headless >> hfo.log &
    # --no-sync >> hfo.log &

    sleep 2
    OFFENSE_AGENT_FILE=$AGENTS_DIR/plastic_dqn_v1/test_player.py
    $PYTHON $OFFENSE_AGENT_FILE --num_opponents=$TOTAL_OPPONENTS \
    --num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_TEST_EPISODES \
    --team_name=$TEAM_NAME --new_model=$NEW_MODEL \
    --starts_fixed_position=$STARTS_FIXED_POSITION --step=$STEP \
    --model_idx=$TEST_IDX --dir=$DIR_NAME --port=$PORT &

    trap "kill -TERM -$$" SIGINT
    wait
  done


  # ** STAGE 3.2: TEST Re-trained models:
  echo "[TEST: STEP $STEP] Re-trained Models"
  NEW_MODEL="false"
  MODEL_BASE="re_${TEAM_NAME}_${STEP}.model"
  NUM_SUB_MODELS=$(ls $DIR_NAME/$MODEL_BASE.* | wc -l)

  for TEST_IDX in $( seq 1 $NUM_SUB_MODELS)
    do

    echo "[TEST: Re ${TEST_IDX}/${NUM_SUB_MODELS}] ${NUM_TEST_EPISODES} episodes"

    MODEL_FILE="${MODEL_BASE}.${test_idx}"
    $HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
     --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
     --offense-on-ball $((-1))  --trials $NUM_TEST_EPISODES --deterministic \
     --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
     --port $PORT \
     --headless >> hfo.log &
    # --no-sync >> hfo.log &

    sleep 2
    OFFENSE_AGENT_FILE=$AGENTS_DIR/plastic_dqn_v1/test_player.py
    $PYTHON $OFFENSE_AGENT_FILE --num_opponents=$TOTAL_OPPONENTS \
    --num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_TEST_EPISODES \
    --team_name=$TEAM_NAME --new_model=$NEW_MODEL \
    --starts_fixed_position=$STARTS_FIXED_POSITION --step=$STEP \
    --model_idx=$TEST_IDX --dir=$DIR_NAME --port=$PORT &

    trap "kill -TERM -$$" SIGINT
    wait
  done

done
