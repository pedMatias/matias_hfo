# !/usr/bin/env python3
# encoding utf-8
import argparse

from agents.plastic_dqn_v1.base.player import Player

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--starts_fixed_position', type=str, default="true")
    
    # Parse arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_episodes
    model_file = args.model_file
    starts_fixed_position = True if args.starts_fixed_position == "true" \
        else False
    
    # Arguments:
    epsilon = 0  # No exploration
    
    print(f"[Run Player] ep={num_episodes}; num_t={num_team}; num_op={num_op}; "
          f"starts_fixed_pos={starts_fixed_position};")
    
    # Player:
    player = Player(num_teammates=num_team, num_opponents=num_op,
                    model_file=model_file, epsilon=epsilon)
    
    # Test Player
    player.play(num_episodes=num_episodes,
                starts_fixed_position=starts_fixed_position)
