#!/bin/bash
killall -9 rcssserver

export PYTHONPATH="/home/matias/Desktop/HFO"
NUMBER_OF_EPISODES=5

./bin/HFO --offense-agents 1 --defense-npcs 1 --offense-on-ball 11  \
--trials $NUMBER_OF_EPISODES --fullstate  --deterministic \
--frames-per-trial 100 --untouched-time 50 --no-sync --no-logging  &
# --no-logging --verbose --headless
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 10
/home/matias/Desktop/HFO/venv/bin/python \
/home/matias/Desktop/HFO/agents/QLearning/q_player.py \
--numOpponents=1 --numTeammates=0 \
--numEpisodes=$NUMBER_OF_EPISODES \
--loadFile="/home/matias/Desktop/HFO/data/q_agent.model.npy" &
echo "Attacker Controller Initialized"


# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait