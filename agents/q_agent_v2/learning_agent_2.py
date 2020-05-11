#!/usr/bin/env python3
# encoding utf-8
import os
import threading

import numpy as np
import argparse
from datetime import datetime as dt

from hfo import SHOOT, MOVE, DRIBBLE, GOAL

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from environement_features.discrete_features import \
    DiscreteHighLevelFeatures
from actions_levels.BaseActions import ActionManager
from environement_features.reward_functions import simple_reward
from matias_hfo import settings
from utils.aux_functions import q_table_variation, get_mean_value_list_by_range
from utils.metrics import BarChart, TwoLineChart


class QLearningAgent:
    name = "q_agent"
    
    def __init__(self, num_states: int, num_actions: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon: float = 1, **kwargs):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_dec = kwargs.get("epsilon_dec") or 0.99
        self.epsilon_end = kwargs.get("epsilon_end") or 0.05
        self.cum_reward = 0
        self.scores = []
        self.test_episodes = []
        self.rewards = []
        self.eps_history = []
        self.lr_history = []
        self.q_table_history = []
        self.old_q_table = np.zeros((num_states, num_actions))
        self.q_table = np.zeros((num_states, num_actions))
        self.save_dir = self._init_instance_directory()
    
    def _init_instance_directory(self):
        now = dt.now().replace(second=0, microsecond=0)
        name_dir = "q_agent_train_" + now.strftime("%Y-%m-%d_%H:%M:%S")
        path = os.path.join(settings.MODELS_DIR, name_dir)
        os.mkdir(path)
        return path
    
    def _check_valid_state(self, state_idx: int):
        if state_idx not in range(self.num_states):
            raise ValueError("State id is wrong! Input:{}; Valid inputs: "
                             "{}".format(state_idx, range(self.num_states)))
    
    def _check_valid_action(self, action_idx: int):
        if action_idx not in range(self.num_actions):
            raise ValueError("Action idx is wrong! Input:{}; Valid inputs: "
                             "{}".format(action_idx, range(self.num_actions)))
    
    def explore_actions(self):
        random_action = np.random.randint(0, self.num_actions)
        return random_action
    
    def exploit_actions(self, state_idx: int) -> int:
        action = np.argmax(self.q_table[state_idx])
        return int(action)
    
    def act(self, state_idx: int):
        if np.random.random() < self.epsilon:  # Explore
            return self.explore_actions()
        else:  # Exploit
            return self.exploit_actions(state_idx)
    
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
        # Update:
        reward = reward if done else \
            reward + self.discount_factor * np.amax(self.q_table[next_state])
        self.q_table[state_idx][action_idx] *= (1 - self.learning_rate)
        self.q_table[state_idx][action_idx] += self.learning_rate * reward
    
    def update_hyper_parameters(self):
        self.learning_rate = self.learning_rate  # TODO update in the future
        # Epsilon:
        self.epsilon = self.epsilon_end if self.epsilon <= self.epsilon_end \
            else self.epsilon * self.epsilon_dec
    
    def reset(self, training: bool = True):
        if training:
            self.old_q_table = self.q_table.copy()
        self.cum_reward = 0
    
    def save_metrics(self, old_q_table: np.ndarray, new_q_table: np.ndarray):
        self.rewards.append(self.cum_reward)
        self.eps_history.append(self.epsilon)
        self.lr_history.append(self.learning_rate)
        self.q_table_history.append(q_table_variation(old_q_table, new_q_table))
    
    def export_metrics(self, training: bool):
        if training:
            rewards = get_mean_value_list_by_range(self.rewards, 10)
            q_vari = get_mean_value_list_by_range(self.q_table_history, 10)
            episodes = list(range(1, len(self.rewards)))
            # Chart 1:
            c1 = TwoLineChart(x_legend="episodes")
            c1.add_first_line_chart(x=episodes, y=self.eps_history,
                                    name="epsilon", y_legend="epsilon")
            c1.add_second_line_chart(x=episodes, y=q_vari,
                                     name="q variation", y_legend="q variation")
            c1.export_as_png(os.path.join(self.save_dir, "epsilon_q_table.png"))
            # Chart 2:
            c2 = TwoLineChart(x_legend="episodes")
            c2.add_first_line_chart(x=episodes, y=self.eps_history,
                                    name="epsilon", y_legend="epsilon")
            c2.add_second_line_chart(x=episodes, y=rewards,
                                     name="reward", y_legend="reward")
            c2.export_as_png(os.path.join(self.save_dir, "epsilon_reward.png"))
            # Chart 3:
            c3 = TwoLineChart(x_legend="episodes")
            c3.add_first_line_chart(x=episodes, y=q_vari,
                                    name="q variation", y_legend="q variation")
            c3.add_second_line_chart(x=episodes, y=rewards,
                                     name="reward", y_legend="reward")
            c3.export_as_png(os.path.join(self.save_dir, "q_table_reward.png"))
        else:
            print("X: ", self.test_episodes)
            print("Y: ", self.scores)
            c1 = BarChart(x_legend="episodes")
            c1.add_bar(x=self.test_episodes, y=self.scores, name="score")
            c1.export_as_png(os.path.join(self.save_dir, "test_score.png"))
    
    def save_model(self):
        file_path = os.path.join(self.save_dir, "q_table")
        np.save(file_path, self.q_table)
            

def test(train_ep: int, num_episodes: int, game_interface: HFOAttackingPlayer,
         features: DiscreteHighLevelFeatures, agent: QLearningAgent,
         actions: ActionManager):
    # Run training using Q-Learning
    score = 0
    agent.test_episodes.append(train_ep)
    for ep in range(num_episodes):
        print('<Test> {}/{}:'.format(ep, num_episodes))
        while game_interface.in_game():
            # Update environment features:
            observation = game_interface.get_state()
            curr_state_id = features.get_state_index(observation)
            has_ball = features.has_ball(observation)
            
            # Act:
            action_idx = agent.exploit_actions(curr_state_id)
            hfo_action = actions.map_action(action_idx)
            
            # Step:
            status, observation = game_interface.step(hfo_action, has_ball)
            agent.cum_reward += reward_function(status)
        print(':: Episode: {}; reward: {}'.format(ep, agent.cum_reward))
        score += 1 if game_interface.status == GOAL else 0
        # Reset player:
        agent.reset(training=False)
        # Game Reset
        game_interface.reset()
    agent.scores.append(score)
    threading.Thread(target=QLearningAgent.export_metrics, args=(agent,
                                                                 False)).start()


def train(num_episodes: int, game_interface: HFOAttackingPlayer,
          features: DiscreteHighLevelFeatures, agent: QLearningAgent,
          actions: ActionManager):
    for ep in range(num_episodes):
        print('<Training> Episode {}/{}:'.format(ep, num_episodes))
        while game_interface.in_game():
            # Update environment features:
            observation = game_interface.get_state()
            curr_state_id = features.get_state_index(observation)
            has_ball = features.has_ball(observation)
            
            # Act:
            action_idx = agent.act(curr_state_id)
            hfo_action = actions.map_action(action_idx)
            
            # Step:
            status, observation = game_interface.step(hfo_action, has_ball)
            reward = reward_function(status)
            agent.cum_reward += reward
            
            # Update environment features:
            prev_state_id = curr_state_id
            curr_state_id = features.get_state_index(observation)
            
            # Update agent
            agent.learn(prev_state_id, action_idx, reward, status,
                        curr_state_id)
        print(':: Episode: {}; reward: {}'.format(ep, agent.cum_reward))
        agent.save_metrics(agent.old_q_table, agent.q_table)
        # Reset player:
        agent.reset()
        agent.update_hyper_parameters()
        # Game Reset
        game_interface.reset()
    threading.Thread(target=QLearningAgent.save_model, args=(agent,)).start()
    threading.Thread(target=QLearningAgent.export_metrics, args=(agent,
                                                                 True)).start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_workouts', type=int, default=10)
    parser.add_argument('--train_batch_dim', type=int, default=500)
    
    args = parser.parse_args()
    agent_id = args.id
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_workouts = args.num_workouts
    train_batch_dim = args.train_batch_dim
    test_batch_dim = 100
    
    print("Starting Training - id={}; num_opponents={}; num_teammates={}; "
          "num_trains={};".format(agent_id, num_op, num_team, num_workouts))
    # Initialize connection with the HFO server
    hfo_interface = HFOAttackingPlayer(agent_id=agent_id,
                                       num_opponents=num_op,
                                       num_teammates=num_team)
    hfo_interface.connect_to_server()
    
    # Agent set-up
    reward_function = simple_reward
    features_manager = DiscreteHighLevelFeatures(num_team, num_op)
    actions_manager = ActionManager([SHOOT, MOVE, DRIBBLE])
    agent = QLearningAgent(num_states=features_manager.get_num_states(),
                           num_actions=actions_manager.get_num_actions(),
                           learning_rate=0.1,
                           discount_factor=0.99,  epsilon=1.0,
                           epsilon_dec=0.9992)
    
    # Run training using Q-Learning
    for i in range(num_workouts):
        print('\n=== Train {}/{}:'.format(i, num_workouts))
        train(num_episodes=train_batch_dim, game_interface=hfo_interface,
              features=features_manager, agent=agent, actions=actions_manager)
        num_train_ep = train_batch_dim + (i * train_batch_dim)
        test(train_ep=num_train_ep, num_episodes=test_batch_dim,
             game_interface=hfo_interface, features=features_manager,
             agent=agent, actions=actions_manager)
