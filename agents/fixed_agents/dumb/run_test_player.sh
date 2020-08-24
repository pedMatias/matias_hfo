#!/bin/bash
killall -9 rcssserver

export PYTHONPATH="/home/matias/Desktop/HFO"

BASE_DIR=/home/matias/Desktop/HFO
HFO=$BASE_DIR/bin/HFO
PYTHON=$BASE_DIR/venv/bin/python
PROJECT_DIR=$BASE_DIR/matias_hfo

export PYTHONPATH=$HFO:$PYTHONPATH
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH

# PORT:
PORT=6010

NUM_GAMES=1

NUM_DEFENSES=1
NUM_DEFENSES_NPCS=0
TOTAL_OPPONENTS=$(($NUM_DEFENSES + $NUM_DEFENSES_NPCS))
echo "TOTAL_OPPONENTS: $TOTAL_OPPONENTS"

NUM_OFFENSES=1
NUM_OFFENSES_NPCS=0
TOTAL_TEAMMATES=$((NUM_OFFENSES + NUM_OFFENSES_NPCS - 1))
echo "TOTAL_TEAMMATES: $TOTAL_TEAMMATES"

# DEFENSE_AGENT_FILE=$BASE_DIR/matias_hfo/agents/goalkeeper/good_teammate.py
DEFENSE_AGENT_FILE=$BASE_DIR/matias_hfo/agents/fixed_teammates/goalkeeper/goalkeeper_v2.py

$HFO --offense-agents $NUM_OFFENSES --offense-npcs $NUM_OFFENSES_NPCS \
--defense-agents $NUM_DEFENSES --defense-npcs $NUM_DEFENSES_NPCS \
 --offense-on-ball 11  --trials $NUM_GAMES --fullstate --no-sync --no-logging \
 --frames-per-trial 100  --port $PORT &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

# OFFENSE_AGENT_FILE=$BASE_DIR/matias_hfo/agents/dumb/dumb_agent.py
# OFFENSE_AGENT_FILE=$BASE_DIR/matias_hfo/agents/dumb/test_agent.py
OFFENSE_AGENT_FILE=$BASE_DIR/matias_hfo/agents/fixed_agents/dumb/test_ball_direccion.py

sleep 2
echo "Connect Offense Player"
$PYTHON $OFFENSE_AGENT_FILE --port=$PORT --num_games=$NUM_GAMES &

sleep 2
echo "Connect Defense Player"
$PYTHON $DEFENSE_AGENT_FILE --port=$PORT --num_episodes=1 &
# .py &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait

