#!/usr/bin/env python3
# encoding utf-8
import json
import os

import numpy as np
import argparse
from datetime import datetime as dt

from hfo import GOAL

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from environement_features import discrete_features_v2, reward_functions
from actions_levels.discrete_actions_v2 import DiscreteActionsV2
from matias_hfo import settings
from utils.aux_functions import q_table_variation


class QLearningAgentV4:
    name = "q_agent"
    EPSILON_VALUES = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    
    def __init__(self, num_states: int, num_actions: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon: float = 0.8, **kwargs):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.cum_reward = 0
        self.scores = []
        self.test_episodes = []
        self.rewards = []
        self.eps_history = []
        self.lr_history = []
        self.visited_states_counter = np.zeros((num_states, num_actions))
        self.q_table_history = []
        self.old_q_table = np.zeros((num_states, num_actions))
        self.q_table = np.zeros((num_states, num_actions))
        self.save_dir = kwargs.get("dir") or self._init_instance_directory()
        self.learning_buffer = []
    
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
    
    def load_q_table(self, load_file):
        print("Loading Q table from file {}".format(load_file))
        self.q_table = np.load(load_file)
    
    def explore_actions(self):
        random_action = np.random.randint(0, self.num_actions)
        return random_action
    
    def exploit_actions(self, state_idx: int) -> int:
        action = np.argmax(self.q_table[state_idx])
        return int(action)
    
    def act(self, state_idx: int):
        # TODO remove, just force the agent to dribble to mid field
        if len(self.learning_buffer) < 4:
            return 4  # Dribble right
        
        if np.random.random() < self.epsilon:  # Explore
            return self.explore_actions()
        else:  # Exploit
            return self.exploit_actions(state_idx)
    
    def _learn(self, state_idx: int, action_idx: int, reward: int, done: bool,
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

        prev_q_value = self.q_table[state_idx][action_idx]
        if done:
            td = reward - prev_q_value
        else:
            max_q_value = np.amax(self.q_table[next_state])
            target_td = reward + (self.discount_factor * max_q_value)
            td = target_td - prev_q_value
        
        self.q_table[state_idx][action_idx] = prev_q_value + \
            self.learning_rate * td
        
        # TODO remove
        #print("pre_q ={} -> (si={}, a={}, r={}, d={}, sf={}) -> "
        #      "new_q={}".format(prev_q_value, state_idx, action_idx, reward,
        #                        done, next_state,
        #                        self.q_table[state_idx][action_idx]))
    
    def learn(self):
        """ The agent only learns from the moment which it has the ball,
        until its final shoot"""
        buffer = self.learning_buffer.copy()
    
        # remove movements without ball
        last_reward = buffer[-1]["r"]
        for i in range(len(self.learning_buffer)-1, -1, -1):
            if buffer[i]["has_ball"]:
                buffer = self.learning_buffer[:i+1]
                break
            else:
                pass
        
        # last reward changed to last action with ball:
        buffer[-1]["r"] = last_reward
        buffer[-1]["done"] = True
        while buffer:
            ep = buffer.pop()
            self._learn(state_idx=ep["st_idx"], action_idx=ep["ac_idx"],
                        reward=ep["r"], next_state=ep["next_st_idx"],
                        done=ep["done"])
    
    def store_ep(self, state_idx: int, action_idx: int, reward: int,
                 next_state_idx: int, has_ball: bool, done: bool):
        entry = {"st_idx": state_idx, "ac_idx": action_idx, "r": reward,
                 "next_st_idx": next_state_idx, "has_ball": has_ball,
                 "done": done}
        self.learning_buffer.append(entry)
    
    def update_hyper_parameters(self, episode: int, num_total_episodes: int):
        self.learning_rate = self.learning_rate  # TODO update in the future
        # Epsilon:
        div = num_total_episodes / len(self.EPSILON_VALUES)
        ep_idx = episode / div
        self.epsilon = self.EPSILON_VALUES[int(ep_idx)]
    
    def reset(self, training: bool = True):
        self.learning_buffer = []
        if training:
            self.old_q_table = self.q_table.copy()
        self.cum_reward = 0
    
    def save_visited_state(self, state_id: int, action_id: int):
        self.visited_states_counter[state_id][action_id] += 1
        
    def save_metrics(self, old_q_table: np.ndarray, new_q_table: np.ndarray):
        self.rewards.append(self.cum_reward)
        self.eps_history.append(self.epsilon)
        self.lr_history.append(self.learning_rate)
        self.q_table_history.append(q_table_variation(old_q_table, new_q_table))
    
    def export_metrics(self, training: bool, actions_name: list):
        """ Saves metrics in Json file"""
        data = {"mode": "train" if training else "test",
                "num_trained_episodes": self.test_episodes,
                "epsilons": self.eps_history,
                "score": self.scores,
                "q_variation": self.q_table_history,
                "actions_label": actions_name,
                "reward": self.rewards,
                "visited_states_counter": self.visited_states_counter.tolist(),
                }
        file_path = os.path.join(self.save_dir, "data.json")
        with open(file_path, 'w+') as fp:
            json.dump(data, fp)
    
    def save_model(self):
        file_path = os.path.join(self.save_dir, "q_table")
        np.save(file_path, self.q_table)
            

def test(train_ep: int, num_episodes: int, game_interface: HFOAttackingPlayer,
         features: discrete_features_v2.DiscreteFeaturesV2,
         agent: QLearningAgentV4, actions: DiscreteActionsV2, reward_funct):
    # Run training using Q-Learning
    score = 0
    agent.test_episodes.append(train_ep)
    for ep in range(num_episodes):
        print('<Test> {}/{}:'.format(ep, num_episodes))
        prev_state_id =-1
        while game_interface.in_game():
            # Update environment features:
            features.update_features(game_interface.get_state())
            curr_state_id = features.get_state_index()
            has_ball = features.has_ball()

            # Act:
            if prev_state_id != curr_state_id:
                print([round(val, 2) for val in agent.q_table[curr_state_id]])
            action_idx = agent.exploit_actions(curr_state_id)
            hfo_action: tuple = actions.map_action_idx_to_hfo_action(
                agent_pos=features.get_pos_tuple(), has_ball=has_ball,
                action_idx=action_idx)
            
            # Step:
            status, observation = game_interface.step(hfo_action, has_ball)
            prev_state_id = curr_state_id
            
            # Save Metrics:
            agent.save_visited_state(curr_state_id, action_idx)
            agent.cum_reward += reward_funct(status)
        print(':: Episode: {}; reward: {}'.format(ep, agent.cum_reward))
        score += 1 if game_interface.status == GOAL else 0
        # Reset player:
        agent.reset(training=False)
        # Game Reset
        game_interface.reset()
    agent.scores.append(score)
    actions_name = [actions_manager.map_action_to_str(i, has_ball=True) for i in
                    range(agent.num_actions)]
    agent.export_metrics(training=False, actions_name=actions_name)


def train(num_episodes: int, game_interface: HFOAttackingPlayer,
          features: discrete_features_v2.DiscreteFeaturesV2,
          agent: QLearningAgentV4, actions: DiscreteActionsV2, reward_funct):
    for ep in range(num_episodes):
        # print('<Training> Episode {}/{}:'.format(ep, num_episodes))
        aux_positions_names = set()
        aux_actions_played = set()
        while game_interface.in_game():
            # Update environment features:
            features.update_features(game_interface.get_state())
            curr_state_id = features.get_state_index()
            has_ball = features.has_ball()
            
            # Act:
            action_idx = agent.act(curr_state_id)
            hfo_action: tuple = actions.map_action_idx_to_hfo_action(
                agent_pos=features.get_pos_tuple(), has_ball=has_ball,
                action_idx=action_idx)
            
            # Step:
            status, observation = game_interface.step(hfo_action, has_ball)
            reward = reward_funct(status)
            
            # Save metrics:
            agent.save_visited_state(curr_state_id, action_idx)
            agent.cum_reward += reward
            aux_positions_names.add(features.get_position_name())
            action_name = actions.map_action_to_str(action_idx, has_ball)
            aux_actions_played.add(action_name)
            
            # Update environment features:
            prev_state_id = curr_state_id
            features.update_features(observation)
            curr_state_id = features.get_state_index()
            agent.store_ep(state_idx=prev_state_id, action_idx=action_idx,
                           reward=reward, next_state_idx=curr_state_id,
                           has_ball=has_ball, done=not game_interface.in_game())
        agent.learn()
        # print(':: Episode: {}; reward: {}; epsilon: {}; positions: {}; '
        #       'actions: {}'.format(ep, agent.cum_reward, agent.epsilon,
        #                            aux_positions_names, aux_actions_played))
        agent.save_metrics(agent.old_q_table, agent.q_table)
        # Reset player:
        agent.reset()
        agent.update_hyper_parameters(episode=ep,
                                      num_total_episodes=num_episodes)
        # Game Reset
        game_interface.reset()
    agent.save_model()
    actions_name = [actions_manager.map_action_to_str(i, has_ball=True) for i in
                    range(agent.num_actions)]
    agent.export_metrics(training=True, actions_name=actions_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--train_mode', type=str, default="train_only",
                        help="Possible Values {}".format(settings.TRAIN_MODES))
    parser.add_argument('--num_train_ep', type=int, default=1000)
    parser.add_argument('--num_test_ep', type=int, default=0)
    
    args = parser.parse_args()
    agent_id = args.id
    num_team = args.num_teammates
    num_op = args.num_opponents
    train_mode = args.train_mode
    num_train_ep = args.num_train_ep
    num_test_ep = args.num_test_ep
    num_episodes = num_train_ep + num_test_ep
    
    print("Starting Training - id={}; num_opponents={}; num_teammates={}; "
          "num_episodes={};".format(agent_id, num_op, num_team, num_episodes))
    # Initialize connection with the HFO server
    hfo_interface = HFOAttackingPlayer(agent_id=agent_id,
                                       num_opponents=num_op,
                                       num_teammates=num_team)
    hfo_interface.connect_to_server()
    
    # Agent set-up
    reward_function = reward_functions.basic_reward
    features_manager = discrete_features_v2.DiscreteFeaturesV2(num_team, num_op)
    actions_manager = DiscreteActionsV2()
    agent = QLearningAgentV4(num_states=features_manager.get_num_states(),
                             num_actions=actions_manager.get_num_actions(),
                             learning_rate=0.1, discount_factor=0.99,
                             epsilon=0.8)
    
    # Run training using Q-Learning
    if train_mode == "train_only":
        print('\n=== Train Mode for {}:'.format(num_train_ep))
        train(num_episodes=num_train_ep, game_interface=hfo_interface,
              features=features_manager, agent=agent,
              actions=actions_manager, reward_funct=reward_function)
    elif train_mode == "alternate":
        print('\n=== Alternating Training Mode. Train {} episodes; Test {} '
              'episodes;'.format(num_train_ep, num_test_ep))
        num_workouts = 10
        train_batch = int(num_train_ep / num_workouts)
        test_batch = int(num_test_ep / num_workouts)
        for i in range(num_workouts):
            train(num_episodes=train_batch, game_interface=hfo_interface,
                  features=features_manager, agent=agent,
                  actions=actions_manager, reward_funct=reward_function)
            train_ep_counter = train_batch + (i * train_batch)
            test(train_ep=train_ep_counter, num_episodes=test_batch,
                 game_interface=hfo_interface, features=features_manager,
                 agent=agent, actions=actions_manager,
                 reward_funct=reward_function)
    elif train_mode == "test_in_the_end":
        print('\n=== Train first, test after Mode. Train {} episodes; '
              'Test {} episodes;'.format(num_train_ep, num_test_ep))
        train(num_episodes=num_train_ep, game_interface=hfo_interface,
              features=features_manager, agent=agent,
              actions=actions_manager, reward_funct=reward_function)
        test(train_ep=num_train_ep, num_episodes=num_test_ep,
             game_interface=hfo_interface, features=features_manager,
             agent=agent, actions=actions_manager,
             reward_funct=reward_function)
    else:
        raise ValueError("Argument train_mode with wrong value")
