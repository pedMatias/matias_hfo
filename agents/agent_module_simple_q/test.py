#!/usr/bin/env python3
# encoding utf-8
import argparse

from agents.agent_module_dqn.player import Player


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--train_mode')
    parser.add_argument('--num_ep', type=int, default=0)
    parser.add_argument('--model_file', type=str, default=None)
    # Parse arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_ep
    model_file = args.model_file
    
    # Start Player:
    player = Player(num_teammates=num_team, num_opponents=num_op)
    # Test Player
    player.agent.epsilon = 0
    player.agent.load_model(model_file)
    av_win_rate = player.test(num_episodes)
    print("Average win rate = {}".format(av_win_rate))
