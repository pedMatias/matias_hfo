#!/bin/bash
killall -9 rcssserver

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
PLAYER_FILE=$BASE_DIR/matias_hfo/multi_agents/plastic/run/run_plastic.py

echo "HFO: ${HFO}"
echo "PYTHON: ${PYTHON}"

export PYTHONPATH=$BASE_DIR:$PYTHONPATH
export PYTHONPATH=$BASE_DIR/matias_hfo:$PYTHONPATH

NUM_DEFENSES=0
NUM_DEFENSES_NPCS=5
TOTAL_DEFENSES=$(($NUM_DEFENSES + $NUM_DEFENSES_NPCS))
TOTAL_OPPONENTS=$TOTAL_DEFENSES
echo "TOTAL_OPPONENTS: $TOTAL_OPPONENTS"

NUM_OFFENSES=4
NUM_OFFENSES_NPCS=0
TOTAL_OFFENSES=$(($NUM_OFFENSES + $NUM_OFFENSES_NPCS))
TOTAL_TEAMMATES=$(($TOTAL_OFFENSES - 1))
echo "TOTAL_TEAMMATES: $TOTAL_TEAMMATES"

# Agent and Teammate Types:
# ("plastic", "memory_bounded", "npc", "correct_policy", "random")
AGENT_TYPE="plastic"
TEAMMATE_TYPE="plastic"

GAME_SET_UP="${TOTAL_OFFENSES}vs${TOTAL_DEFENSES}"
MODELS_DIR=$BASE_DIR/matias_hfo/models/${GAME_SET_UP}
AUX_NAME="${NUM_OFFENSES}+${NUM_OFFENSES_NPCS}_game_w_${TEAMMATE_TYPE}_$(date +"%Y-%m-%dT%T")"
# AUX_NAME="2+0_game_2020-12-05T09:40:04"
# AUX_NAME="${TOTAL_OFFENSES}agents_history${HISTORY_LEN}_$(date +"%Y-%m-%dT%T")"

# METRICS_DIR=$MODELS_DIR/metrics/${AGENT_TYPE}/${AUX_NAME}
METRICS_DIR="home/matias/Desktop/HFO/matias_hfo/models/4vs5/metrics/plastic/w_plastic_agent/stochastic/4+0_game_w_plastic_2020-12-07T13:00:29"
echo $METRICS_DIR
mkdir $METRICS_DIR

TEAMS_NAMES=("aut" "axiom" "cyrus" "gliders" "helios")

PORT=6100
# 200 * 5 = 1000 Trials
for t in {55..250}
do
  NUM_EPISODES=50
  TEAM_NAME="base"  # ${TEAMS_NAMES[i]}
  TEAM_METRICS_DIR=${METRICS_DIR}/${TEAM_NAME}
  mkdir $TEAM_METRICS_DIR

  echo "[TEAM: ${TEAM_NAME}] STARTING..."
  $HFO --offense-team $TEAM_NAME --offense-agents $NUM_OFFENSES \
  --offense-npcs $NUM_OFFENSES_NPCS --defense-agents $NUM_DEFENSES \
  --defense-npcs $NUM_DEFENSES_NPCS --offense-on-ball $((-1))  \
  --trials $NUM_EPISODES --deterministic --fullstate --no-logging \
  --frames-per-trial 500 --untouched-time 300 --port $PORT \
  --headless &
  # --no-sync &

  # Main Agent:
  MODEL_TYPE="stochastic"  # ("stochastic", "adversarial")
  PREFIX="t${t}_${AGENT_TYPE}1"
  USE_WEBSERVICE="true"
  HISTORY_LEN=0
  PLOT_METRICS="false"
  SAVE="true"

  sleep 3
  $PYTHON $PLAYER_FILE --team_name=$TEAM_NAME --num_teammates=$TOTAL_TEAMMATES \
  --num_opponents=$TOTAL_OPPONENTS --num_episodes=$NUM_EPISODES \
  --history_len=$HISTORY_LEN --models_dir=$MODELS_DIR --port=$PORT \
  --save=$SAVE --model_type=$MODEL_TYPE --metrics_dir=$TEAM_METRICS_DIR \
  --prefix=$PREFIX --plot=$PLOT_METRICS --use_webservice=$USE_WEBSERVICE \
  --agent_type=$AGENT_TYPE &

  # Plastic Agent 1:
  MODEL_TYPE="stochastic"
  HISTORY_LEN=0
  PREFIX="t${t}_${AGENT_TYPE}2"
  USE_WEBSERVICE="true"
  PLOT_METRICS="false"
  SAVE="false"

  sleep 3
  $PYTHON $PLAYER_FILE --team_name=$TEAM_NAME --num_teammates=$TOTAL_TEAMMATES \
  --num_opponents=$TOTAL_OPPONENTS --num_episodes=$NUM_EPISODES \
  --history_len=$HISTORY_LEN --models_dir=$MODELS_DIR --port=$PORT \
  --save=$SAVE --model_type=$MODEL_TYPE --metrics_dir=$TEAM_METRICS_DIR \
  --prefix=$PREFIX --plot=$PLOT_METRICS --use_webservice=$USE_WEBSERVICE \
  --agent_type=$TEAMMATE_TYPE &

  # Plastic Agent 2:
  MODEL_TYPE="stochastic"
  HISTORY_LEN=0
  PREFIX="t${t}_${AGENT_TYPE}2"
  USE_WEBSERVICE="true"
  PLOT_METRICS="false"
  SAVE="false"

  sleep 3
  $PYTHON $PLAYER_FILE --team_name=$TEAM_NAME --num_teammates=$TOTAL_TEAMMATES \
  --num_opponents=$TOTAL_OPPONENTS --num_episodes=$NUM_EPISODES \
  --history_len=$HISTORY_LEN --models_dir=$MODELS_DIR --port=$PORT \
  --save=$SAVE --model_type=$MODEL_TYPE --metrics_dir=$TEAM_METRICS_DIR \
  --prefix=$PREFIX --plot=$PLOT_METRICS --use_webservice=$USE_WEBSERVICE \
  --agent_type=$TEAMMATE_TYPE &

  # Plastic Agent 3:
  MODEL_TYPE="stochastic"
  HISTORY_LEN=0
  PREFIX="t${t}_${AGENT_TYPE}3"
  USE_WEBSERVICE="true"
  PLOT_METRICS="false"
  SAVE="false"

  sleep 3
  $PYTHON $PLAYER_FILE --team_name=$TEAM_NAME --num_teammates=$TOTAL_TEAMMATES \
  --num_opponents=$TOTAL_OPPONENTS --num_episodes=$NUM_EPISODES \
  --history_len=$HISTORY_LEN --models_dir=$MODELS_DIR --port=$PORT \
  --save=$SAVE --model_type=$MODEL_TYPE --metrics_dir=$TEAM_METRICS_DIR \
  --prefix=$PREFIX --plot=$PLOT_METRICS --use_webservice=$USE_WEBSERVICE \
  --agent_type=$TEAMMATE_TYPE &


  trap "kill -TERM -$$" SIGINT
  wait

  sleep 5
done
