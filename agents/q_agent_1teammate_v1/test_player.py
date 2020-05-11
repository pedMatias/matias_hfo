#!/usr/bin/env python3
# encoding utf-8
import argparse

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from environement_features.discrete_features_1teammate_v1 import \
    DiscreteFeatures1TeammateV1
from environement_features.reward_functions import basic_reward
from actions_levels.discrete_actions_1teammate_v1 import \
    DiscreteActions1TeammateV1
from agents.q_agent_1teammate_v1.qagent import QLearningAgent
from agents.q_agent_1teammate_v1.train_player_w_non_static import test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_ep', type=int, default=0)
    parser.add_argument('--load_file', type=str, default=None)
    
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_ep
    load_file = args.load_file
    
    print("Q Table file: ", load_file)
    print("Starting Test - num_opponents={}; num_teammates={}; "
          "num_episodes={};".format(num_op, num_team, num_episodes))
    # Initialize connection with the HFO server
    hfo_interface = HFOAttackingPlayer(num_opponents=num_op,
                                       num_teammates=num_team)
    hfo_interface.connect_to_server()
    
    # Agent set-up
    reward_function = basic_reward
    features_manager = DiscreteFeatures1TeammateV1(num_team, num_op)
    actions_manager = DiscreteActions1TeammateV1()
    agent = QLearningAgent(num_states=features_manager.get_num_states(),
                           num_actions=actions_manager.get_num_actions())
    agent.load_q_table(load_file)
    
    # Run training using Q-Learning
    av_reward = test(num_episodes=num_episodes, agent=agent,
                     game_interface=hfo_interface, features=features_manager,
                     actions=actions_manager, reward_funct=reward_function)
    print("Av reward = {}".format(av_reward))
