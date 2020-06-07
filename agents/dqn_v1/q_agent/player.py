#!/usr/bin/env python3
# encoding utf-8
import random

import numpy as np
from hfo import GOAL, IN_GAME

import settings
from agents.base.hfo_attacking_player import HFOAttackingPlayer
from agents.dqn_v1.q_agent.q_agent import QAgent
from environement_features.reward_functions import basic_reward
from agents.dqn_v1.actions.simple import Actions
from agents.dqn_v1.features.discrete_features import DiscFeatures1Teammate

STARTING_POSITIONS = {"TOP LEFT": (-0.5, -0.7), "TOP RIGHT": (0.4, -0.7),
                      "MID LEFT": (-0.5, 0.0), "MID RIGHT": (0.4, 0.0),
                      "BOTTOM LEFT": (-0.5, 0.7), "BOTTOM RIGHT": (0.4, 0.7)}


class Player:
    def __init__(self, num_opponents: int, num_teammates: int,
                 port: int = 6000):
        # Game Interface:
        self.game_interface = HFOAttackingPlayer(num_opponents=num_opponents,
                                                 num_teammates=num_teammates,
                                                 port=port)
        self.game_interface.connect_to_server()
        # Features Interface:
        self.features = DiscFeatures1Teammate(num_op=num_opponents,
                                              num_team=num_teammates)
        # Actions Interface:
        self.actions = Actions()
        # Agent instance:
        self.agent = QAgent(num_features=self.features.num_features,
                            num_actions=self.actions.get_num_actions(),
                            learning_rate=0.1, discount_factor=0.9, epsilon=0.8)
    
    def get_reward(self, status: int) -> int:
        return basic_reward(status)
    
    def set_starting_game_conditions(self, game_interface: HFOAttackingPlayer,
                                     features: DiscFeatures1Teammate,
                                     start_with_ball: bool = True,
                                     start_pos: tuple = None):
        """
        Set starting game conditions. Move for initial position, for example
        """
        if not start_pos:
            pos_name, start_pos = random.choice(
                list(STARTING_POSITIONS.items()))
        if start_with_ball:
            # Move to starting position:
            self.actions.dribble_to_pos(start_pos, features, game_interface)
        else:
            if self.features.has_ball():
                self.actions.kick_to_pos((0, 0), features, game_interface)
            # Move to starting position:
            self.actions.move_to_pos(start_pos, features, game_interface)
        # Informs the other players that it is ready to start:
        game_interface.hfo.say(settings.PLAYER_READY_MSG)

    def test(self, num_episodes: int, start_with_ball: bool = True) -> float:
        """
        @param num_episodes: number of episodes to run
        @param start_with_ball: flag
        @return: (float) the win rate
        """
        starting_pos_list = list(STARTING_POSITIONS.values())
        # metrics variables:
        _num_wins = 0
        for ep in range(num_episodes):
            # Check if server still running:
            self.game_interface.check_server_is_up()
            # Update features:
            self.features.update_features(self.game_interface.get_state())
            # Set up gaming conditions:
            self.set_starting_game_conditions(
                game_interface=self.game_interface, features=self.features,
                start_pos=starting_pos_list[ep % len(starting_pos_list)],
                start_with_ball=start_with_ball)
            print("\nNEW TEST [{}]".format(
                starting_pos_list[ep % len(starting_pos_list)]))

            # Start learning loop
            status = IN_GAME
            prev_action_idx = None
            while self.game_interface.in_game():
                if self.features.has_ball():
                    # Update environment features:
                    features_array = self.features.get_features().copy()
        
                    # Act:
                    action_idx = self.agent.exploit_actions(features_array)
                    if prev_action_idx != action_idx:
                        print("ACTION:: {}".format(
                            self.actions.map_action_to_str(action_idx)))
                    prev_action_idx = action_idx
                    self.actions.execute_action(
                        action_idx=action_idx,
                        features=self.features,
                        game_interface=self.game_interface)
                else:
                    if prev_action_idx != -1:
                        print("ACTION:: MOVE!!")
                    prev_action_idx = -1
                    status = self.actions.no_ball_action(
                        features=self.features,
                        game_interface=self.game_interface)

            # Update auxiliar variables:
            if self.game_interface.scored_goal() or status == GOAL:
                print("[GOAL]")
                _num_wins += 1
            else:
                print("[FAIL]")
            # Game Reset
            self.game_interface.reset()
        avr_win_rate = _num_wins / num_episodes
        print("[TEST: Summary] WIN rate = {};".format(avr_win_rate))
        return avr_win_rate

    def train(self, num_train_episodes: int, num_total_train_ep: int,
              start_with_ball: bool = True):
        """
        @param num_train_episodes: number of episodes to train in this iteration
        @param num_total_train_ep: number total of episodes to train
        @param start_with_ball: bool
        @raise ServerDownError
        @return: (QLearningAgentV5) the agent
        """
        # metrics variables:
        _num_wins = 0
        _sum_epsilons = 0
        for ep in range(num_train_episodes):
            # Check if server still running:
            self.game_interface.check_server_is_up()
            # Update features:
            self.features.update_features(self.game_interface.get_state())
            # Go to origin position:
            self.set_starting_game_conditions(
                game_interface=self.game_interface, features=self.features,
                start_with_ball=start_with_ball)
            
            # Start learning loop
            status = IN_GAME
            episode_buffer = list()
            while self.game_interface.in_game():
                # Has Ball:
                if self.features.has_ball():
                    # Update environment features:
                    features_array = self.features.get_features().copy()
                
                    # Act:
                    action_idx = self.agent.act(features_array)
                    status = self.actions.execute_action(
                        action_idx=action_idx,
                        features=self.features,
                        game_interface=self.game_interface)
    
                    # Every step we update replay memory and train main network
                    done = not self.game_interface.in_game()
                    # Store transition:
                    # (obs, action, reward, new obs, done?)
                    transition = np.array(
                        [features_array, action_idx, self.get_reward(status),
                         self.features.get_features(), done])
                    episode_buffer.append(transition)
                # No ball:
                else:
                    status = self.actions.no_ball_action(
                        features=self.features,
                        game_interface=self.game_interface)
            if self.game_interface.scored_goal() or status == GOAL:
                _num_wins += 1
                reward = self.get_reward(GOAL)
            else:
                reward = self.get_reward(status)
            self.agent.store_episode(episode_buffer, reward=reward)
            # Train:
            self.agent.train(terminal_state=True)
            # Update auxiliar variables:
            _sum_epsilons += self.agent.epsilon
            # Update Agent:
            self.agent.restart(num_total_train_ep)
            # Game Reset
            self.game_interface.reset()
        print("[TRAIN: Summary] WIN rate = {}; AVR epsilon = {}".format(
            _num_wins / num_train_episodes, _sum_epsilons / num_train_episodes))
