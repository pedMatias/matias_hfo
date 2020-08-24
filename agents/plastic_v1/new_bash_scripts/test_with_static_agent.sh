#!/bin/bash
killall -9 rcssserver

export PYTHONPATH=/home/matias/Desktop/HFO:$PYTHONPATH
export PYTHONPATH=/home/matias/Desktop/HFO/matias_hfo:$PYTHONPATH
echo $PYTHONPATH

PORT=6000

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
MODULE_DIR=$BASE_DIR/matias_hfo/agents
DATA_DIR=$BASE_DIR/matias_hfo/data

# Test config:
OFFENSE_AGENT_FILE=$MODULE_DIR/plastic_v1/dqn_test.py
# Agent Model:
MODEL=/home/matias/Desktop/HFO/matias_hfo/data/offline_4000ep_plasticFeaturesF_plasticSimplexActionsA_2020-06-28_2/agent_model

NUM_EPISODES=12

GOALKEEPER_TYPE="good_goalkeeper"  # dumb_goalkeeper, good_goalkeeper, helios
TEAMMATE_TYPE="helios"  # static_teammate, good_teammate, helios

if [ "$GOALKEEPER_TYPE" == "helios" ]; then
  NUM_DEFENSES=0
  NUM_DEFENSES_NPCS=1
else
  NUM_DEFENSES=1
  NUM_DEFENSES_NPCS=0
fi
TOTAL_DEFENSES=$(($NUM_DEFENSES + $NUM_DEFENSES_NPCS))
TOTAL_OPPONENTS=$TOTAL_DEFENSES
echo "TOTAL_OPPONENTS: $TOTAL_OPPONENTS"

if [ "$TEAMMATE_TYPE" == "helios" ]; then
  NUM_OFFENSES=1
  NUM_OFFENSES_NPCS=1
else
  NUM_OFFENSES=2
  NUM_OFFENSES_NPCS=0
fi
TOTAL_OFFENSES=$(($NUM_OFFENSES + $NUM_OFFENSES_NPCS))
TOTAL_TEAMMATES=$(($TOTAL_OFFENSES - 1))
echo "TOTAL_TEAMMATES: $TOTAL_TEAMMATES"

$HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
 --defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
 --offense-on-ball 6  --trials $NUM_EPISODES --deterministic --fullstate \
 --frames-per-trial 200 --no-logging \
 --no-sync  &
# --headless &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 3
echo "Connect to Main player"
$PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
--num_teammates=$TOTAL_TEAMMATES --num_ep=$NUM_EPISODES --model_file=$MODEL &
echo "PLayer connected"

if [ "$TEAMMATE_TYPE" != "helios" ]; then
  sleep 3
  echo "[TEAMMATE] Connect to Teammate Player"
  TEAMMATE_AGENT_FILE=$MODULE_DIR/fixed_agents/fixed_teammate/$TEAMMATE_TYPE.py
  echo $TEAMMATE_AGENT_FILE
  $PYTHON $TEAMMATE_AGENT_FILE  --num_episodes=$NUM_EPISODES --port=$PORT \
  --num_opponents=$TOTAL_OPPONENTS --num_teammates=$TOTAL_TEAMMATES &
fi

if [ "$GOALKEEPER_TYPE" != "helios" ]; then
  sleep 3
  echo "[GOALKEEPER] Connect Goalkeeper Player"
  GOALKEEPER_AGENT_FILE=$MODULE_DIR/fixed_agents/goalkeeper/$GOALKEEPER_TYPE.py
  echo $GOALKEEPER_AGENT_FILE
  $PYTHON $GOALKEEPER_AGENT_FILE  --num_episodes=$NUM_EPISODES --port=$PORT \
  --num_offenses=$TOTAL_OFFENSES --num_defenses=$(($TOTAL_DEFENSES-1)) &
fi

# .py &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait