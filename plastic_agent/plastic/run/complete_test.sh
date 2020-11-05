#!/bin/bash
killall -9 rcssserver

PORT=6000

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
MODELS_DIR=$BASE_DIR/matias_hfo/models
PLAYER_FILE=$BASE_DIR/matias_hfo/plastic_agent/plastic/run/run_plastic.py

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

METRICS_DIR=$MODELS_DIR/metrics/$(date +"%Y-%m-%dT%T")
echo $METRICS_DIR
mkdir $METRICS_DIR

TEAMS_NAMES=("axiom" "cyrus" "gliders" "aut" "helios")

for i in {0..4}
do
  SAVE="true"
  NUM_EPISODES=1000
  MEMORY_BOUNDED="true"
  HISTORY_LEN=5
  TEAM_NAME=${TEAMS_NAMES[i]}

  echo "[TEAM: ${TEAM_NAME}] STARTING..."
  $HFO --offense-team $TEAM_NAME --offense-agents $NUM_OFFENSES \
  --offense-npcs $NUM_OFFENSES_NPCS --defense-agents $NUM_DEFENSES \
  --defense-npcs $NUM_DEFENSES_NPCS --offense-on-ball $((-1))  \
  --trials $NUM_EPISODES --deterministic --fullstate --no-logging \
  --frames-per-trial 500 --untouched-time 300 --port $PORT \
  --headless &
  # --no-sync &

  sleep 2
  $PYTHON $PLAYER_FILE --team_name=$TEAM_NAME --num_teammates=$TOTAL_TEAMMATES \
  --num_opponents=$TOTAL_OPPONENTS --num_episodes=$NUM_EPISODES \
  --memory_bounded=$MEMORY_BOUNDED --history_len=$HISTORY_LEN \
  --models_dir=$MODELS_DIR --port=$PORT --save=$SAVE \
  --metrics_dir=$METRICS_DIR&

  trap "kill -TERM -$$" SIGINT
  wait

done
