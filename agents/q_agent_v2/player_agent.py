#!/usr/bin/env python3
# encoding utf-8
import numpy as np
import argparse

from hfo import SHOOT, MOVE, DRIBBLE

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from environement_features.discrete_features import \
    DiscreteHighLevelFeatures
from actions_levels.BaseActions import ActionManager
from environement_features.reward_functions import simple_reward
from utils.utils import plot_learning
import settings


class QPlayerAgent:
    def __init__(self, num_states: int, num_actions: int, num_games: int,
                 load_file: str):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_games = num_games
        self.score = 0
        self.scores = []
        self.q_table = np.load(load_file)
    
    def act(self, state_idx: int):
        action = np.argmax(self.q_table[state_idx])
        return action
    
    def _save_metrics(self, episode: int, produce_graph: bool = False):
        self.scores.append(self.score)
        eps_history = [0 for _ in range(episode)]
        if produce_graph and episode > 0:
            file_name = "q_player_agent_{}_{}_{}.png".format(
                self.num_games, self.num_states, self.num_actions)
            file_path = settings.IMAGES_DIR + file_name
            x = [i + 1 for i in range(episode)]
            plot_learning(x, self.scores, eps_history, file_path)
    
    def reset(self, episode: int, produce_graph: bool = False):
        if episode == 0:
            pass
        else:
            self._save_metrics(episode, produce_graph)
            self.score = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--load_file', type=str, default=None)
    
    args = parser.parse_args()
    agent_id = args.id
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_episodes
    load_file = args.load_file
    
    print("Starting Player - id={}; num_opponents={}; num_teammates={}; "
          "num_episodes={}; load_file={};".format(agent_id, num_op, num_team,
                                                  num_episodes, load_file))
    # Initialize connection with the HFO server
    hfo_interface = HFOAttackingPlayer(agent_id=agent_id,
                                       num_opponents=args.num_opponents,
                                       num_teammates=args.num_teammates)
    hfo_interface.connect_to_server()

    # Reward Function
    reward_function = simple_reward
    
    # Get number of features and actions
    features_manager = DiscreteHighLevelFeatures(num_team, num_op)
    actions_manager = ActionManager([SHOOT, MOVE, DRIBBLE])
    
    # Initialize a Q-Learning Agent
    agent = QPlayerAgent(num_states=features_manager.get_num_states(),
                         num_actions=actions_manager.get_num_actions(),
                         num_games=num_episodes,
                         load_file=load_file)
    
    for i in range(num_episodes):
        agent.reset(i, produce_graph=True)
        observation = hfo_interface.reset()
        # Update environment features:
        curr_state_id = features_manager.get_state_index(observation)
        has_ball = features_manager.has_ball(observation)
        
        while hfo_interface.in_game():
            action_idx = agent.act(curr_state_id)
            hfo_action = actions_manager.map_action(action_idx)
            
            status, observation = hfo_interface.step(hfo_action, has_ball)
            agent.score += reward_function(status)
            
            # Update environment features:
            curr_state_id = features_manager.get_state_index(observation)
            has_ball = features_manager.has_ball(observation)