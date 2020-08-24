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

# ************************************* STAGE 1 ********************************
# Train with a static_teammate, against a dumb_goalkeeper
STAGE=1
echo "** STAGE $STAGE **"

STARTS_WITH_BALL="true"
STARTS_FIXED_POSITION="true"
GOALKEEPER_TYPE="dumb_goalkeeper"  # dumb_goalkeeper, good_goalkeeper, helios
TEAMMATE_TYPE="static_teammate"  # static_teammate, good_teammate, helios

NUM_EPISODES_LIST=(5000 5000 5000 5000 5000)
EPSILONS_LIST=(1 0.6 0.4 0.2 0.1)

#for i in {1..4}
#do
#  SUB_STAGE=$STAGE.$i
#  NUM_EPISODES=${NUM_EPISODES_LIST[i]}
#  EPSILON=${EPSILONS_LIST[i]}
#
#  # STAGE 1.1: Explore for the first time
#  $HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
#   --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
#   --offense-on-ball $((-1))  --trials $NUM_EPISODES --deterministic \
#   --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
#   --port $PORT --headless >> hfo.log &
#
#  sleep 2
#  echo "Connect to Main player"
#  OFFENSE_AGENT_FILE=$AGENTS_DIR/offline_plastic_v2/explore_player.py
#  $PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
#  --num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_EPISODES \
#  --teammate_type=$TEAMMATE_TYPE --opponent_type=$GOALKEEPER_TYPE \
#  --starts_with_ball=$STARTS_WITH_BALL --epsilon=$EPSILON \
#  --starts_fixed_position=$STARTS_FIXED_POSITION --port=$PORT \
#  --dir=$DIR_NAME --stage=$SUB_STAGE &
#
#  sleep 2
#  echo "[TEAMMATE] Connect to Teammate Player"
#  TEAMMATE_AGENT_FILE=$AGENTS_DIR/fixed_agents/fixed_teammate/$TEAMMATE_TYPE.py
#  echo $TEAMMATE_AGENT_FILE
#  $PYTHON $TEAMMATE_AGENT_FILE  --num_episodes=$NUM_EPISODES --port=$PORT \
#  --num_opponents=$TOTAL_OPPONENTS --num_teammates=$TOTAL_TEAMMATES &
#
#  sleep 2
#  echo "[GOALKEEPER] Connect Goalkeeper Player"
#  GOALKEEPER_AGENT_FILE=$AGENTS_DIR/fixed_agents/goalkeeper/$GOALKEEPER_TYPE.py
#  echo $GOALKEEPER_AGENT_FILE
#  $PYTHON $GOALKEEPER_AGENT_FILE  --num_episodes=$NUM_EPISODES --port=$PORT \
#  --num_offenses=$TOTAL_OFFENSES --num_defenses=$(($TOTAL_DEFENSES-1)) &
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#
#  # ** STAGE 1.2 **: TRAIN OFFLINE:
#  TRAIN_SCRIPT=$AGENTS_DIR/offline_plastic_v2/train_offline.py
#  echo "Start Train"
#  $PYTHON $TRAIN_SCRIPT --num_opponents=$TOTAL_OPPONENTS \
#    --num_teammates=$TOTAL_TEAMMATES --dir=$DIR_NAME --stage=$SUB_STAGE \
#    --num_train_rep=20 &
#  echo "PLayer connected"
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#
#  # ** STAGE 1.3 **: TEST:
#  CHOOSE_BEST_SUB_MODEL="true"
#  NUM_SUB_MODELS=$(ls $DIR_NAME/agent_model_$SUB_STAGE.* | wc -l)
#
#  for test_idx in $( seq 1 $NUM_SUB_MODELS)
#    do
#    TEST_EPISODES=50
#    echo "[STAGE $STAGE: TEST] $test_idx / $NUM_SUB_MODELS EPISODES"
#    TOTAL_TEST_EPISODES=$(($NUM_SUB_MODELS * $TEST_EPISODES))
#    $HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
#     --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
#     --offense-on-ball $((-1))  --trials $TEST_EPISODES --deterministic \
#     --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
#     --port $PORT \
#     --headless >> hfo.log &
#    # --no-sync >> hfo.log &
#
#    sleep 2
#    echo "Connect to Main player"
#    OFFENSE_AGENT_FILE=$AGENTS_DIR/offline_plastic_v2/test_player.py
#    $PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
#    --num_teammates=$TOTAL_TEAMMATES --num_episodes=$TEST_EPISODES \
#    --dir=$DIR_NAME --stage=$SUB_STAGE --starts_with_ball=$STARTS_WITH_BALL \
#    --starts_fixed_position=$STARTS_FIXED_POSITION \
#    --test_iter=$test_idx --choose_best_sub_model=$CHOOSE_BEST_SUB_MODEL &
#
#    sleep 2
#    echo "[TEAMMATE] Connect to Teammate Player"
#    TEAMMATE_AGENT_FILE=$AGENTS_DIR/fixed_agents/fixed_teammate/$TEAMMATE_TYPE.py
#    echo $TEAMMATE_AGENT_FILE
#    $PYTHON $TEAMMATE_AGENT_FILE  --num_episodes=$TEST_EPISODES --port=$PORT \
#    --num_opponents=$TOTAL_OPPONENTS --num_teammates=$TOTAL_TEAMMATES &
#
#    sleep 2
#    echo "[GOALKEEPER] Connect Goalkeeper Player"
#    GOALKEEPER_AGENT_FILE=$AGENTS_DIR/fixed_agents/goalkeeper/$GOALKEEPER_TYPE.py
#    echo $GOALKEEPER_AGENT_FILE
#    $PYTHON $GOALKEEPER_AGENT_FILE  --num_episodes=$TEST_EPISODES --port=$PORT \
#    --num_offenses=$TOTAL_OFFENSES --num_defenses=$(($TOTAL_DEFENSES-1)) &
#
#    trap "kill -TERM -$$" SIGINT
#    wait
#  done
#
#done


# ************************************* STAGE 2 ********************************
# Train with a static_teammate, against a helios goalkeeper
STAGE=2
echo "** STAGE $STAGE **"

STARTS_WITH_BALL="false"
STARTS_FIXED_POSITION="false"
GOALKEEPER_TYPE="helios"  # dumb_goalkeeper, good_goalkeeper, helios
TEAMMATE_TYPE="static_teammate"  # static_teammate, good_teammate, helios

NUM_DEFENSES=0
NUM_DEFENSES_NPCS=1
TOTAL_DEFENSES=$(($NUM_DEFENSES + $NUM_DEFENSES_NPCS))
TOTAL_OPPONENTS=$TOTAL_DEFENSES
echo "TOTAL_OPPONENTS: $TOTAL_OPPONENTS"

NUM_EPISODES_LIST=(5000 5000 5000)
EPSILONS_LIST=(0.6 0.3 0.1)

#for i in {0..2}
#do
#  SUB_STAGE=$STAGE.$i
#  NUM_EPISODES=${NUM_EPISODES_LIST[i]}
#  EPSILON=${EPSILONS_LIST[i]}
#
#  # STAGE 2.1: Explore for the first time
#  $HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
#   --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
#   --offense-on-ball $((-1))  --trials $NUM_EPISODES --deterministic \
#   --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
#   --port $PORT --headless >> hfo.log &
#
#  sleep 2
#  echo "Connect to Main player"
#  OFFENSE_AGENT_FILE=$AGENTS_DIR/offline_plastic_v2/explore_player.py
#  $PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
#  --num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_EPISODES \
#  --teammate_type=$TEAMMATE_TYPE --opponent_type=$GOALKEEPER_TYPE \
#  --starts_with_ball=$STARTS_WITH_BALL --epsilon=$EPSILON \
#  --starts_fixed_position=$STARTS_FIXED_POSITION --port=$PORT \
#  --dir=$DIR_NAME --stage=$SUB_STAGE &
#
#  sleep 2
#  echo "[TEAMMATE] Connect to Teammate Player"
#  TEAMMATE_AGENT_FILE=$AGENTS_DIR/fixed_agents/fixed_teammate/$TEAMMATE_TYPE.py
#  echo $TEAMMATE_AGENT_FILE
#  $PYTHON $TEAMMATE_AGENT_FILE  --num_episodes=$NUM_EPISODES --port=$PORT \
#  --num_opponents=$TOTAL_OPPONENTS --num_teammates=$TOTAL_TEAMMATES &
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#
#  # ** STAGE 2.2 **: TRAIN OFFLINE:
#  TRAIN_SCRIPT=$AGENTS_DIR/offline_plastic_v2/train_offline.py
#  echo "Start Train"
#  $PYTHON $TRAIN_SCRIPT --num_opponents=$TOTAL_OPPONENTS \
#    --num_teammates=$TOTAL_TEAMMATES --dir=$DIR_NAME --stage=$SUB_STAGE \
#    --num_train_rep=20 &
#  echo "PLayer connected"
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#
#  # ** STAGE 2.3 **: TEST:
#  CHOOSE_BEST_SUB_MODEL="true"
#  NUM_SUB_MODELS=$(ls $DIR_NAME/agent_model_$SUB_STAGE.* | wc -l)
#
#  for test_idx in $( seq 1 $NUM_SUB_MODELS)
#    do
#    TEST_EPISODES=50
#    echo "[STAGE $STAGE: TEST] $test_idx / $NUM_SUB_MODELS EPISODES"
#    TOTAL_TEST_EPISODES=$(($NUM_SUB_MODELS * $TEST_EPISODES))
#    $HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
#     --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
#     --offense-on-ball $((-1))  --trials $TEST_EPISODES --deterministic \
#     --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
#     --port $PORT \
#     --headless >> hfo.log &
#    # --no-sync >> hfo.log &
#
#    sleep 2
#    echo "Connect to Main player"
#    OFFENSE_AGENT_FILE=$AGENTS_DIR/offline_plastic_v2/test_player.py
#    $PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
#    --num_teammates=$TOTAL_TEAMMATES --num_episodes=$TEST_EPISODES \
#    --dir=$DIR_NAME --stage=$SUB_STAGE --starts_with_ball=$STARTS_WITH_BALL \
#    --starts_fixed_position=$STARTS_FIXED_POSITION \
#    --test_iter=$test_idx --choose_best_sub_model=$CHOOSE_BEST_SUB_MODEL &
#
#    sleep 2
#    echo "[TEAMMATE] Connect to Teammate Player"
#    TEAMMATE_AGENT_FILE=$AGENTS_DIR/fixed_agents/fixed_teammate/$TEAMMATE_TYPE.py
#    echo $TEAMMATE_AGENT_FILE
#    $PYTHON $TEAMMATE_AGENT_FILE  --num_episodes=$TEST_EPISODES --port=$PORT \
#    --num_opponents=$TOTAL_OPPONENTS --num_teammates=$TOTAL_TEAMMATES &
#
#    trap "kill -TERM -$$" SIGINT
#    wait
#  done
#
#done


# ************************************* STAGE 3 ********************************
# Train with a helios, against a helios goalkeeper
STAGE=3
echo "** STAGE $STAGE **"

STARTS_WITH_BALL="false"
STARTS_FIXED_POSITION="false"
GOALKEEPER_TYPE="helios"  # dumb_goalkeeper, good_goalkeeper, helios
TEAMMATE_TYPE="helios"  # static_teammate, good_teammate, helios

NUM_DEFENSES=0
NUM_DEFENSES_NPCS=1
TOTAL_DEFENSES=$(($NUM_DEFENSES + $NUM_DEFENSES_NPCS))
TOTAL_OPPONENTS=$TOTAL_DEFENSES
echo "TOTAL_OPPONENTS: $TOTAL_OPPONENTS"

NUM_OFFENSES=1
NUM_OFFENSES_NPCS=1
TOTAL_OFFENSES=$(($NUM_OFFENSES + $NUM_OFFENSES_NPCS))
TOTAL_TEAMMATES=$(($TOTAL_OFFENSES - 1))
echo "TOTAL_TEAMMATES: $TOTAL_TEAMMATES"

NUM_EPISODES_LIST=(10000 10000 10000 10000 10000 10000)
EPSILONS_LIST=(0.6 0.5 0.4 0.3 0.2 0.1)

#for i in {0..5}
#do
#  SUB_STAGE=$STAGE.$i
#  NUM_EPISODES=${NUM_EPISODES_LIST[i]}
#  EPSILON=${EPSILONS_LIST[i]}
#
#  # STAGE 3.1: Explore for the first time
#  $HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
#   --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
#   --offense-on-ball $((-1))  --trials $NUM_EPISODES --deterministic \
#   --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
#   --port $PORT --headless >> hfo.log &
#
#  sleep 2
#  echo "Connect to Main player"
#  OFFENSE_AGENT_FILE=$AGENTS_DIR/offline_plastic_v2/explore_player.py
#  $PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
#  --num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_EPISODES \
#  --teammate_type=$TEAMMATE_TYPE --opponent_type=$GOALKEEPER_TYPE \
#  --starts_with_ball=$STARTS_WITH_BALL --epsilon=$EPSILON \
#  --starts_fixed_position=$STARTS_FIXED_POSITION --port=$PORT \
#  --dir=$DIR_NAME --stage=$SUB_STAGE &
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#
#  # ** STAGE 3.2 **: TRAIN OFFLINE:
#  TRAIN_SCRIPT=$AGENTS_DIR/offline_plastic_v2/train_offline.py
#  echo "Start Train"
#  $PYTHON $TRAIN_SCRIPT --num_opponents=$TOTAL_OPPONENTS \
#    --num_teammates=$TOTAL_TEAMMATES --dir=$DIR_NAME --stage=$SUB_STAGE \
#    --num_train_rep=25 &
#  echo "PLayer connected"
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#
#  # ** STAGE 3.3 **: TEST:
#  CHOOSE_BEST_SUB_MODEL="true"
#  NUM_SUB_MODELS=$(ls $DIR_NAME/agent_model_$SUB_STAGE.* | wc -l)
#
#  for test_idx in $( seq 1 $NUM_SUB_MODELS)
#    do
#    TEST_EPISODES=500
#    echo "[STAGE $STAGE: TEST] $test_idx / $NUM_SUB_MODELS EPISODES"
#    TOTAL_TEST_EPISODES=$(($NUM_SUB_MODELS * $TEST_EPISODES))
#    $HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
#     --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
#     --offense-on-ball $((-1))  --trials $TEST_EPISODES --deterministic \
#     --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
#     --port $PORT \
#     --headless >> hfo.log &
#    # --no-sync >> hfo.log &
#
#    sleep 2
#    echo "Connect to Main player"
#    OFFENSE_AGENT_FILE=$AGENTS_DIR/offline_plastic_v2/test_player.py
#    $PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
#    --num_teammates=$TOTAL_TEAMMATES --num_episodes=$TEST_EPISODES \
#    --dir=$DIR_NAME --stage=$SUB_STAGE --starts_with_ball=$STARTS_WITH_BALL \
#    --starts_fixed_position=$STARTS_FIXED_POSITION \
#    --test_iter=$test_idx --choose_best_sub_model=$CHOOSE_BEST_SUB_MODEL &
#
#    trap "kill -TERM -$$" SIGINT
#    wait
#  done
#
#done


# ************************************* STAGE 4 *******************************
# Train with an helios teammate, against 2 helios defenses
# Neural net 128_128_128_128
STAGE=4
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

NUM_EPISODES_LIST=(50000 25000 25000 25000 25000 25000 25000 25000 25000 25000)
EPSILONS_LIST=(0.5 0.4 0.3 0.2 0.1 0.1 0.1 0.1 0.1 0.1)

#for i in {4..9}
#do
#  SUB_STAGE=$STAGE.$i
#  NUM_EPISODES=${NUM_EPISODES_LIST[i]}
#  EPSILON=${EPSILONS_LIST[i]}
#
#  # STAGE 4.1: Explore for the first time
#  $HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
#   --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
#   --offense-on-ball $((-1))  --trials $NUM_EPISODES --deterministic \
#   --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
#   --port $PORT --headless >> hfo.log &
#  sleep 2
#
#  echo "Connect to Main player"
#  OFFENSE_AGENT_FILE=$AGENTS_DIR/offline_plastic_v2/explore_player.py
#  $PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
#  --num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_EPISODES \
#  --teammate_type=$TEAMMATE_TYPE --opponent_type=$GOALKEEPER_TYPE \
#  --starts_with_ball=$STARTS_WITH_BALL --epsilon=$EPSILON \
#  --starts_fixed_position=$STARTS_FIXED_POSITION --port=$PORT \
#  --dir=$DIR_NAME --stage=$SUB_STAGE &
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#  # ** STAGE 4.2 **: TRAIN OFFLINE:
#  TRAIN_SCRIPT=$AGENTS_DIR/offline_plastic_v2/train_offline.py
#  echo "Start Train"
#  $PYTHON $TRAIN_SCRIPT --num_opponents=$TOTAL_OPPONENTS \
#    --num_teammates=$TOTAL_TEAMMATES --dir=$DIR_NAME --stage=$SUB_STAGE \
#    --replay_buffer_size=200000 --save_all="false" &
#  echo "PLayer connected"
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#
#  # ** STAGE 4.3 **: TEST:
#  CHOOSE_BEST_SUB_MODEL="true"
#  NUM_SUB_MODELS=$(ls $DIR_NAME/agent_model_$SUB_STAGE.* | wc -l)
#
#  for test_idx in $( seq 1 $NUM_SUB_MODELS)
#  do
#    TEST_EPISODES=500
#    echo "[STAGE $STAGE: TEST] $test_idx / $NUM_SUB_MODELS EPISODES"
#    $HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
#     --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
#     --offense-on-ball $((-1))  --trials $TEST_EPISODES --deterministic \
#     --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
#     --port $PORT \
#     --headless >> hfo.log &
#    # --no-sync >> hfo.log &
#
#    sleep 3
#    OFFENSE_AGENT_FILE=$AGENTS_DIR/offline_plastic_v2/test_player.py
#    $PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
#    --num_teammates=$TOTAL_TEAMMATES --num_episodes=$TEST_EPISODES \
#    --dir=$DIR_NAME --stage=$SUB_STAGE --starts_with_ball=$STARTS_WITH_BALL \
#    --starts_fixed_position=$STARTS_FIXED_POSITION --test_iter=$test_idx \
#    --choose_best_sub_model=$CHOOSE_BEST_SUB_MODEL &
#
#    trap "kill -TERM -$$" SIGINT
#    wait
#  done
#
#done

# ************************************* STAGE 5 *******************************
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

NUM_EPISODES_LIST=(25000 25000 25000)
EPSILONS_LIST=(0.1 0.1 0.1)

for i in {2..9}
do
  SUB_STAGE=$STAGE.$i
  NUM_EPISODES=${NUM_EPISODES_LIST[i]}
  EPSILON=${EPSILONS_LIST[i]}

#  # STAGE 5.1: Explore for the first time
#  $HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
#   --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
#   --offense-on-ball $((-1))  --trials $NUM_EPISODES --deterministic \
#   --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
#   --port $PORT --headless >> hfo.log &
#  sleep 2
#
#  echo "Connect to Main player"
#  OFFENSE_AGENT_FILE=$AGENTS_DIR/offline_plastic_v2/explore_player.py
#  $PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
#  --num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_EPISODES \
#  --teammate_type=$TEAMMATE_TYPE --opponent_type=$GOALKEEPER_TYPE \
#  --starts_with_ball=$STARTS_WITH_BALL --epsilon=$EPSILON \
#  --starts_fixed_position=$STARTS_FIXED_POSITION --port=$PORT \
#  --dir=$DIR_NAME --stage=$SUB_STAGE &
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#  # ** STAGE 5.2 **: TRAIN OFFLINE:
#  TRAIN_SCRIPT=$AGENTS_DIR/offline_plastic_v2/train_offline.py
#  echo "Start Train"
#  $PYTHON $TRAIN_SCRIPT --num_opponents=$TOTAL_OPPONENTS \
#    --num_teammates=$TOTAL_TEAMMATES --dir=$DIR_NAME --stage=$SUB_STAGE \
#    --replay_buffer_size=500000 --save_all="true" &
#  echo "PLayer connected"
#
#  trap "kill -TERM -$$" SIGINT
#  wait


  # ** STAGE 5.3 **: TEST:
  CHOOSE_BEST_SUB_MODEL="true"
  NUM_SUB_MODELS=$(ls $DIR_NAME/agent_model_$SUB_STAGE.* | wc -l)

  for test_idx in $( seq 1 $NUM_SUB_MODELS)
  do
    TEST_EPISODES=100
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

done


# ************************************* STAGE 6 *******************************
# Train with an helios teammate, against 2 helios defenses
# Neural net 128_128_128_128
STAGE=6
echo "** STAGE $STAGE **"

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

STARTS_WITH_BALL="false"
STARTS_FIXED_POSITION="false"
GOALKEEPER_TYPE="helios"  # dumb_goalkeeper, good_goalkeeper, helios
TEAMMATE_TYPE="dqa"  # static_teammate, good_teammate, helios

NUM_EPISODES_LIST=(25000 25000 25000)
EPSILONS_LIST=(0.1 0.1 0.1)

for i in {2..9}
do
  SUB_STAGE=$STAGE.$i
  NUM_EPISODES=${NUM_EPISODES_LIST[i]}
  EPSILON=${EPSILONS_LIST[i]}

#  # STAGE 6.1: Explore for the first time
#  $HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
#   --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
#   --offense-on-ball $((-1))  --trials $NUM_EPISODES --deterministic \
#   --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
#   --port $PORT --headless >> hfo.log &
#  sleep 2
#
#  echo "Connect to Main player"
#  OFFENSE_AGENT_FILE=$AGENTS_DIR/offline_plastic_v2/explore_player.py
#  $PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
#  --num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_EPISODES \
#  --teammate_type=$TEAMMATE_TYPE --opponent_type=$GOALKEEPER_TYPE \
#  --starts_with_ball=$STARTS_WITH_BALL --epsilon=$EPSILON \
#  --starts_fixed_position=$STARTS_FIXED_POSITION --port=$PORT \
#  --dir=$DIR_NAME --stage=$SUB_STAGE &
#
#  trap "kill -TERM -$$" SIGINT
#  wait
#
#  # ** STAGE 6.2 **: TRAIN OFFLINE:
#  TRAIN_SCRIPT=$AGENTS_DIR/offline_plastic_v2/train_offline.py
#  echo "Start Train"
#  $PYTHON $TRAIN_SCRIPT --num_opponents=$TOTAL_OPPONENTS \
#    --num_teammates=$TOTAL_TEAMMATES --dir=$DIR_NAME --stage=$SUB_STAGE \
#    --replay_buffer_size=500000 --save_all="true" &
#  echo "PLayer connected"
#
#  trap "kill -TERM -$$" SIGINT
#  wait


  # ** STAGE 6.3 **: TEST:
  CHOOSE_BEST_SUB_MODEL="true"
  NUM_SUB_MODELS=$(ls $DIR_NAME/agent_model_$SUB_STAGE.* | wc -l)

  for test_idx in $( seq 1 $NUM_SUB_MODELS)
  do
    TEST_EPISODES=100
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

done