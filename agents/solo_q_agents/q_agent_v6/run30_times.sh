#!/bin/bash

for i in {0..5}
  do
     echo " "
     echo "---------------------Repeating loop $i /30 times-------------------"
     source /home/matias/Desktop/HFO/matias_hfo/agents/q_agent_v6/train_agent.sh
     sleep 10
 done

