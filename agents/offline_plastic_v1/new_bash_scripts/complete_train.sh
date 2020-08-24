#!/bin/bash
killall -9 rcssserver

export PYTHONPATH=/home/matias/Desktop/HFO:$PYTHONPATH
export PYTHONPATH=/home/matias/Desktop/HFO/matias_hfo:$PYTHONPATH
echo $PYTHONPATH

PORT=6010

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
MODULE_DIR=$BASE_DIR/matias_hfo/agents

# Train config:
NUM_EPISODES=6000
echo "Episodes: $NUM_EPISODES"

OFFENSE_AGENT_FILE=$MODULE_DIR/plastic_v1/explore_player.py

GOALKEEPER_TYPE="good_goalkeeper"  # dumb_goalkeeper, good_goalkeeper, helios
TEAMMATE_TYPE="static_teammate"  # static_teammate, good_teammate, helios


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
 --offense-on-ball $((-1))  --trials $NUM_EPISODES --deterministic --fullstate \
 --no-logging --frames-per-trial 500 --untouched-time 300 --port $PORT \
 --headless >> hfo.log &
# --no-sync >> hfo.log &

sleep 2
echo "Connect to Main player"
$PYTHON $OFFENSE_AGENT_FILE  --num_opponents=$TOTAL_OPPONENTS \
--num_teammates=$TOTAL_TEAMMATES --num_episodes=$NUM_EPISODES \
--teammate_type=$TEAMMATE_TYPE --opponent_type=$GOALKEEPER_TYPE \
--starts_with_ball="false" --starts_fixed_position="true" --port=$PORT &
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


trap "kill -TERM -$$" SIGINT
wait

