#!/usr/bin/env python3
# encoding utf-8
import numpy as np


class QLearningAgentTest:
    name = "q_agent"
    EPSILON_VALUES = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    
    def __init__(self, num_states: int, num_actions: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon: float = 0.8):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        # used to learn process:
        self.trained_eps = 0
        self.learning_buffer = []
        self.q_table = np.zeros((num_states, num_actions))
    
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
        if np.random.random() < self.epsilon:  # Explore
            return self.explore_actions()
        else:  # Exploit
            return self.exploit_actions(state_idx)
    
    def learn_episode(self, state_idx: int, action_idx: int, reward: int,
                      done: bool, next_state: int):
        """
        Called at each loop iteration when the agent is learning. It should
        implement the learning procedure.
        @param state_idx: Old State id - the id that identifies the state
        @param action_idx: Action id - range(0, self.num_actions)
        @param reward: reward
        @param done: Game ended
        @param next_state: New State id - the id that identifies the state):
        """
        prev_q_value = self.q_table[state_idx][action_idx].copy()
        if done:
            td = reward - prev_q_value
        else:
            max_q_value = np.amax(self.q_table[next_state])
            target_td = reward + (self.discount_factor * max_q_value)
            td = target_td - prev_q_value
        
        self.q_table[state_idx][action_idx] = prev_q_value + \
                                              (self.learning_rate * td)

        # print("pre_q ={} -> (si={}, a={}, r={}, d={}, sf={}) && next_q-> "
        #       "new_q={}".format(prev_q_value, state_idx, action_idx, reward,
        #                         done, next_state,
        #                         self.q_table[state_idx][action_idx]))
    
    def learn_buffer(self, last_reward: int):
        """ The agent only learns from the moment which it has the ball,
        until its final shoot"""
        # Inc number of trained episodes:
        self.trained_eps += 1
        # Update last learning entry:
        buffer = self.learning_buffer.copy()
        buffer[-1]["r"] = last_reward
        buffer[-1]["done"] = True
        # last reward changed to last action with ball:
        while buffer:
            ep = buffer.pop()
            self.learn_episode(state_idx=ep["st_idx"], action_idx=ep["ac_idx"],
                               reward=ep["r"], next_state=ep["next_st_idx"],
                               done=ep["done"])
        # reset learning buffer:
        self.learning_buffer = []
    
    def store_ep(self, state_idx: int, action_idx: int, reward: int,
                 next_state_idx: int, has_ball: bool, done: bool):
        # Store entry:
        entry = {"st_idx": state_idx, "ac_idx": action_idx, "r": reward,
                 "next_st_idx": next_state_idx, "has_ball": has_ball,
                 "done": done}
        self.learning_buffer.append(entry)
    
    def update_hyper_parameters(self, num_total_episodes: int):
        self.learning_rate = self.learning_rate  # TODO update in the future
        # Epsilon:
        epsilon_idx = int((self.trained_eps * len(self.EPSILON_VALUES)) /
                          num_total_episodes)
        if epsilon_idx >= len(self.EPSILON_VALUES):
            self.epsilon = self.EPSILON_VALUES[-1]
        else:
            self.epsilon = self.EPSILON_VALUES[int(epsilon_idx)]