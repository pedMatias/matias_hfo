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

DIR_NAME=$DATA_DIR/complete_train_v1

# Train with a static_teammate, against a good_goalkeeper. Starts without ball
MODEL_STAGE="9.1"
echo "** STAGE $MODEL_STAGE **"

STARTS_WITH_BALL="false"
STARTS_FIXED_POSITION="false"
TEAMMATE_WAIT_FOR_AGENT="false"
GOALKEEPER_TYPE="helios"  # dumb_goalkeeper, good_goalkeeper, helios
TEAMMATE_TYPE="helios"  # static_teammate, good_teammate, helios

TEST_EPISODES=10

$HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
 --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
 --offense-on-ball -1  --trials $TEST_EPISODES --deterministic \
 --fullstate --no-logging --frames-per-trial 500 --untouched-time 300 \
 --port $PORT \
 --no-sync  >> hfo.log &
# --headless >> hfo.log &

sleep 3
OFFENSE_AGENT_FILE=$AGENTS_DIR/offline_plastic_v1/test_player.py
$PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
--num_teammates=$TOTAL_TEAMMATES --num_episodes=$TEST_EPISODES \
--dir=$DIR_NAME --stage=$MODEL_STAGE --starts_with_ball=$STARTS_WITH_BALL \
--starts_fixed_position=$STARTS_FIXED_POSITION --save="false" &

#sleep 1
#TEAMMATE_AGENT_FILE=$AGENTS_DIR/fixed_agents/fixed_teammate/$TEAMMATE_TYPE.py
#echo $TEAMMATE_AGENT_FILE
#$PYTHON $TEAMMATE_AGENT_FILE  --num_episodes=$TEST_EPISODES --port=$PORT \
#--num_opponents=$TOTAL_OPPONENTS --num_teammates=$TOTAL_TEAMMATES \
#--starts_fixed_position=$STARTS_FIXED_POSITION \
#--wait_for_teammate=$TEAMMATE_WAIT_FOR_AGENT &
#
#sleep 1
#GOALKEEPER_AGENT_FILE=$AGENTS_DIR/fixed_agents/goalkeeper/$GOALKEEPER_TYPE.py
#echo $GOALKEEPER_AGENT_FILE
#$PYTHON $GOALKEEPER_AGENT_FILE  --num_episodes=$TEST_EPISODES --port=$PORT \
#--num_offenses=$TOTAL_OFFENSES --num_defenses=$(($TOTAL_DEFENSES-1)) &

trap "kill -TERM -$$" SIGINT
wait