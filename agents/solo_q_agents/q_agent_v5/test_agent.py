#!/usr/bin/env python3
# encoding utf-8
import argparse

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from environement_features import discrete_features_v2, reward_functions
from actions_levels.discrete_actions_v5 import DiscreteActionsV5
from agents.solo_q_agents.q_agent_v5 import learning_agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_ep', type=int, default=0)
    parser.add_argument('--load_file', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    
    args = parser.parse_args()
    agent_id = args.id
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_ep
    load_file = args.load_file
    save_dir = args.save_dir
    
    print("Q Table file: ", load_file)
    print("Starting Test - num_opponents={}; num_teammates={}; "
          "num_episodes={};".format(num_op, num_team, num_episodes))
    # Initialize connection with the HFO server
    hfo_interface = HFOAttackingPlayer(agent_id=agent_id,
                                       num_opponents=num_op,
                                       num_teammates=num_team)
    hfo_interface.connect_to_server()
    
    # Agent set-up
    reward_function = reward_functions.basic_reward
    features_manager = discrete_features_v2.DiscreteFeaturesV2(num_team, num_op)
    actions_manager = DiscreteActionsV5(
        origin_pos=learning_agent.ORIGIN_POSITIONS["MID LEFT"])
    agent = learning_agent.QLearningAgentV5(
        num_states=features_manager.get_num_states(),
        num_actions=actions_manager.get_num_actions(),
        dir=save_dir)
    agent.load_q_table(load_file)
    
    # Run training using Q-Learning
    av_reward = learning_agent.test(num_episodes=num_episodes, agent=agent,
                                    game_interface=hfo_interface,
                                    features=features_manager,
                                    actions=actions_manager,
                                    reward_funct=reward_function)
    print("Av reward = {}".format(av_reward))