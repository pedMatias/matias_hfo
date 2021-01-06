#!/usr/bin/hfo_env python3
# encoding utf-8
import argparse

from agents.agent_module_dqn.deep_agent import DQNAgent
from actions_levels.action_module import DiscreteActionsModule
from agents.agent_module_dqn.features.discrete_features import \
    DiscreteFeatures1Teammate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--train_mode')
    parser.add_argument('--num_ep', type=int, default=0)
    parser.add_argument('--load_file', type=str, default=None)
    # Parse arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_ep
    load_file = args.load_file
    
    actions = DiscreteActionsModule()
    features = DiscreteFeatures1Teammate(num_op=num_op, num_team=num_team)
    # Start Player:
    agent = DQNAgent(features.get_num_features(), actions.get_num_actions())
    # Test Player
    agent.epsilon = 0
    agent.load_model(load_file)
    av_win_rate = player.test(num_episodes)
    print("Average win rate = {}".format(av_win_rate))
