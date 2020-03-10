#!/usr/bin/env python3
# encoding utf-8

from .hfo_attacking_player import HFOAttackingPlayer
from environement_features.base import \
    BaseHighLevelState
from actions_levels.BaseActions import ActionManager
from environement_features.reward_functions import simple_reward
import argparse


class BaseLearningAgent:
    def __init__(self, num_features: int, num_actions: int,
                 learning_rate: float, discount_factor: float, epsilon: float,
                 save_file: str = None):
        self.num_actions = num_actions
        self.num_features = num_features
        self.lr = learning_rate
        self.df = discount_factor
        self.ep = epsilon
        self.save_file = save_file
        self.scores = []
        self.eps_history = []

    def act(self, features: list):
        """ Called at each loop iteration to choose and execute an action.
        Returns:
            None
        """
        raise NotImplementedError

    def learn(self):
        """ Called at each loop iteration when the agent is learning. It should
        implement the learning procedure.
        Returns:
            None
        """
        raise NotImplementedError

    def set_experience(self, state, action, reward, status, next_state):
        raise NotImplementedError

    def set_learning_rate(self, learning_rate):
        raise NotImplementedError

    def set_epsilon(self, epsilon):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def update_hyper_parameters(self, num_taken_actions, episode_number):
        raise NotImplementedError
        # self.set_epsilon(self.ep)
        # self.set_learning_rate(self.set_learning_rate())
    
    def save_model(self):
        raise NotImplementedError


# TODO """ The code bellow is an example """
"""
def main():
    ''' The code bellow is an example '''
    # TODO raise NotImplementedError
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--saveFile', type=str, default="q_agent.model")
    
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_episodes
    saving_file = args.saveFile

    # Initialize connection with the HFO server
    hfo_interface = HFOAttackingPlayer(num_opponents=args.num_opponents,
                                       num_teammates=args.num_teammates,
                                       agent_id=args.id)
    hfo_interface.connect_to_server()
    
    # Reward Function
    reward_function = simple_reward
    
    # Get number of features and actions
    features_manager = HighLevelState(num_team=num_team, num_op=num_op)
    n_actions = ActionManager.get_num_actions()

    # Initialize a Q-Learning Agent
    agent = BaseLearningAgent(num_features=features_manager.get_num_features(),
                              num_actions=n_actions,
                              learning_rate=0.1, discount_factor=0.99,
                              epsilon=1.0, save_file=saving_file)

    # Run training using Q-Learning
    num_taken_actions = 0
    for episode in range(num_episodes):
        observation = hfo_interface.reset()
        curr_features = features_manager.get_features(observation)
        has_ball = features_manager.has_ball(observation)

        while hfo_interface.in_game():
            agent.update_hyper_parameters(num_taken_actions, episode)
            action = agent.act(curr_features)
            num_taken_actions += 1
            
            status, observation = hfo_interface.step(action, has_ball)
            reward = reward_function(status)
            
            # Update environment features:
            prev_features = curr_features.copy()
            curr_features = features_manager.get_features(observation)
            has_ball = features_manager.has_ball(observation)
            
            # Update agent
            agent.set_experience(prev_features, action, reward, status,
                                 curr_features)
            update = agent.learn()

"""

