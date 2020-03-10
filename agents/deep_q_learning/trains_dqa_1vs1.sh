#!/bin/bash
killall -9 rcssserver

export PYTHONPATH="/home/matias/Desktop/HFO"

./bin/HFO --offense-agents 1 --defense-npcs 1 --offense-on-ball 1  \
--trials 1000  &  # --no-sync
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 5
#/home/matias/Desktop/HFO/venv/bin/python \
#  /home/matias/Desktop/HFO/agents/deep_q_learning/player_dqn.py
# .py &
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
#trap "kill -TERM -$$" SIGINT
#wait

