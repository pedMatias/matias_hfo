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

DIR_NAME=$DATA_DIR/complete_train_v2
# DIR_NAME=$DATA_DIR/complete_train_complete_actions$(date +"%Y-%m-%dT%T")
mkdir $DIR_NAME

# ************************************* STAGE 4 *******************************
# Train with an helios teammate, against 2 helios defenses
# Neural net 128_128_128_128
STAGE=5
echo "** STAGE $STAGE **"

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

STARTS_WITH_BALL="false"
STARTS_FIXED_POSITION="false"
GOALKEEPER_TYPE="helios"  # dumb_goalkeeper, good_goalkeeper, helios
TEAMMATE_TYPE="helios"  # static_teammate, good_teammate, helios

SUB_STAGE=5.0

## ** STAGE 4.2 **: TRAIN OFFLINE:
#TRAIN_SCRIPT=$AGENTS_DIR/offline_plastic_v2/train_offline.py
#echo "Start Train"
#$PYTHON $TRAIN_SCRIPT --num_opponents=$TOTAL_OPPONENTS \
#  --num_teammates=$TOTAL_TEAMMATES --dir=$DIR_NAME --stage=$SUB_STAGE \
#  --replay_buffer_size=2000000 --save_all="true" &
#echo "PLayer connected"
#
#trap "kill -TERM -$$" SIGINT
#wait


# ** STAGE 4.3 **: TEST:
CHOOSE_BEST_SUB_MODEL="true"
NUM_SUB_MODELS=$(ls $DIR_NAME/agent_model_$SUB_STAGE.* | wc -l)

for test_idx in $( seq 1 $NUM_SUB_MODELS)
do
  TEST_EPISODES=50
  echo "[STAGE $STAGE: TEST] $test_idx / $NUM_SUB_MODELS EPISODES"
  $HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
   --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
   --offense-on-ball $((-1))  --trials $TEST_EPISODES --deterministic \
   --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
   --port $PORT \
   --headless >> hfo.log &
  # --no-sync >> hfo.log &

  sleep 3
  OFFENSE_AGENT_FILE=$AGENTS_DIR/offline_plastic_v2/test_player.py
  $PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
  --num_teammates=$TOTAL_TEAMMATES --num_episodes=$TEST_EPISODES \
  --dir=$DIR_NAME --stage=$SUB_STAGE --starts_with_ball=$STARTS_WITH_BALL \
  --starts_fixed_position=$STARTS_FIXED_POSITION --test_iter=$test_idx \
  --choose_best_sub_model=$CHOOSE_BEST_SUB_MODEL &

  trap "kill -TERM -$$" SIGINT
  wait
done
exit