#!/bin/bash
killall -9 rcssserver

export PYTHONPATH="/home/matias/Desktop/HFO"

NUMBER_OF_EPISODES=1000

./bin/HFO --offense-agents 1 --defense-npcs 1 --offense-on-ball 11  \
--trials $NUMBER_OF_EPISODES --fullstate  --deterministic \
--frames-per-trial 1000 --untouched-time 100 --no-logging  &
# --no-logging --verbose --no-sync
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 5
/home/matias/Desktop/HFO/venv/bin/python \
/home/matias/Desktop/HFO/agents/QLearning/q_agent.py \
--numOpponents=1 --numTeammates=0 --numEpisodes=$NUMBER_OF_EPISODES \
--saveFile="/home/matias/Desktop/HFO/data/q_agent.model" &
echo "Attacker Controller Initialized"


# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait