#!/usr/bin/hfo_env python3
# encoding utf-8
import numpy as np
import argparse
from datetime import datetime as dt

from hfo import SHOOT, MOVE, DRIBBLE

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from environement_features.discrete_features import \
    DiscreteHighLevelFeatures
from actions_levels.BaseActions import ActionManager
from environement_features.reward_functions import simple_reward
from utils.aux_functions import plot_learning
from matias_hfo import settings


class QLearningAgent:
    name = "q_agent"
    
    def __init__(self, num_states: int, num_actions: int, num_games: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon: float = 1, save_file: str = None, **kwargs):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_games = num_games
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_dec = kwargs.get("epsilon_dec") or 0.996
        self.epsilon_end = kwargs.get("epsilon_end") or 0.05
        self.save_file = save_file if save_file else self._gen_file_name()
        self.score = 0
        self.scores = []
        self.eps_history = []
        self.q_table = np.zeros((num_states, num_actions))
    
    def _gen_file_name(self):
        return "qlearning_agent_{}_{}_{}_{}".format(self.num_games,
                                                    self.num_states,
                                                    self.num_actions,
                                                    self.learning_rate)
    
    def _check_valid_state(self, state_idx: int):
        if state_idx not in range(self.num_states):
            raise ValueError("State id is wrong! Input:{}; Valid inputs: "
                             "{}".format(state_idx, range(self.num_states)))
    
    def _check_valid_action(self, action_idx: int):
        if action_idx not in range(self.num_actions):
            raise ValueError("Action idx is wrong! Input:{}; Valid inputs: "
                             "{}".format(action_idx, range(self.num_actions)))
    
    def act(self, state_idx: int):
        if np.random.random() < self.epsilon:  # Explore
            random_action = np.random.randint(0, self.num_actions)
            return random_action
        else:  # Exploit
            action = np.argmax(self.q_table[state_idx])
            return action
    
    def learn(self, state_idx: int, action_idx: int, reward: int, done: bool,
              next_state: int):
        """
        Called at each loop iteration when the agent is learning. It should
        implement the learning procedure.
        @param state_idx: Old State id - the id that identifies the state
        @param action_idx: Action id - range(0, self.num_actions)
        @param reward: reward
        @param done: Game ended
        @param next_state: New State id - the id that identifies the state):
        """
        self._check_valid_state(state_idx), self._check_valid_state(next_state)
        self._check_valid_action(action_idx)
        # Save Score:
        self.score += reward
        # Update:
        reward = reward if done else \
            reward + self.discount_factor * np.amax(self.q_table[next_state])
        self.q_table[state_idx][action_idx] *= (1 - self.learning_rate)
        self.q_table[state_idx][action_idx] += self.learning_rate * reward
    
    def _update_hyper_parameters(self):
        self.learning_rate = self.learning_rate  # TODO update in the future
        # Epsilon:
        self.epsilon = self.epsilon_end if self.epsilon <= self.epsilon_end \
            else self.epsilon * self.epsilon_dec
    
    def reset(self, episode: int):
        if episode > 0:
            self.scores.append(self.score)
            self.eps_history.append(self.epsilon)
            self._update_hyper_parameters()
            self.score = 0
    
    def save_plot(self, episode: int, file_name: str = None):
        if file_name is None:
            file_name = settings.PLOT_FILE_NAME_FORMAT.format(
                agent_type=self.name,
                num_episodes=self.num_games,
                date=dt.now().isoformat()
            )
        file_path = settings.IMAGES_DIR + file_name + ".png"
        x = [i + 1 for i in range(episode)]
        plot_learning(x, self.scores, self.eps_history, file_path)
    
    def save_model(self):
        file_path = settings.MODELS_DIR + self.save_file
        np.save(file_path, self.q_table)
    
    def save(self, episode: int, produce_graph: bool):
        self.save_model()
        if produce_graph:
            self.save_plot(episode, self.save_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--save_file', type=str, default=None)
    
    args = parser.parse_args()
    agent_id = args.id
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_episodes
    saving_file = args.save_file
    
    print("Starting Training - id={}; num_opponents={}; num_teammates={}; "
          "num_episodes={}; saveFile={};".format(agent_id, num_op, num_team,
                                                 num_episodes, saving_file))
    # Initialize connection with the HFO server
    hfo_interface = HFOAttackingPlayer(agent_id=agent_id,
                                       num_opponents=num_op,
                                       num_teammates=num_team)
    hfo_interface.connect_to_server()
    
    # Reward Function
    reward_function = simple_reward
    
    # Get number of features and actions
    features_manager = DiscreteHighLevelFeatures(num_team, num_op)
    actions_manager = ActionManager([SHOOT, MOVE, DRIBBLE])
    
    # Initialize a Q-Learning Agent
    agent = QLearningAgent(num_states=features_manager.get_num_states(),
                           num_actions=actions_manager.get_num_actions(),
                           learning_rate=0.1,
                           discount_factor=0.99,  epsilon=1.0,
                           num_games=num_episodes,
                           save_file=saving_file)
    
    # Run training using Q-Learning
    for i in range(num_episodes):
        print('\n=== Episode {}/{}:'.format(i, num_episodes))
        agent.reset(i)
        observation = hfo_interface.reset()
        # Update environment features:
        curr_state_id = features_manager.get_state_index(observation)
        has_ball = features_manager.has_ball(observation)
        
        while hfo_interface.in_game():
            action_idx = agent.act(curr_state_id)
            hfo_action = actions_manager.map_action(action_idx)
            
            status, observation = hfo_interface.step(hfo_action, has_ball)
            reward = reward_function(status)
            
            # Update environment features:
            prev_state_id = curr_state_id
            curr_state_id = features_manager.get_state_index(observation)
            has_ball = features_manager.has_ball(observation)
            
            # Update agent
            agent.learn(prev_state_id, action_idx, reward, status,
                        curr_state_id)

        print(':: Episode: {}; Score: {}'.format(i, agent.score))
        if i % 1000 == 0 and i > 0:
            agent.save(i, produce_graph=True)
