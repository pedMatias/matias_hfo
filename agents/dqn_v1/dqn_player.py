#!/usr/bin/env python3
# encoding utf-8
import random

import numpy as np
from hfo import GOAL, IN_GAME, CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS

import settings
from agents.utils import ServerDownError
from agents.base.hfo_attacking_player import HFOAttackingPlayer
from agents.dqn_v1.deep_agent import DQNAgent
from agents.dqn_v1.actions.simple import Actions
from agents.dqn_v1.features.plastic_features import PlasticFeatures

STARTING_POSITIONS = {"TOP LEFT": (-0.5, -0.7), "TOP RIGHT": (0.4, -0.7),
                      "MID LEFT": (-0.5, 0.0), "MID RIGHT": (0.4, 0.0),
                      "BOTTOM LEFT": (-0.5, 0.7), "BOTTOM RIGHT": (0.4, 0.7)}


class Player:
    def __init__(self, num_opponents: int, num_teammates: int,
                 port: int = 6000, online: bool = True):
        # Game Interface:
        self.game_interface = HFOAttackingPlayer(num_opponents=num_opponents,
                                                 num_teammates=num_teammates,
                                                 port=port)
        if online:
            self.game_interface.connect_to_server()
        # Features Interface:
        self.features = PlasticFeatures(num_op=num_opponents,
                                        num_team=num_teammates)
        # Actions Interface:
        self.actions = Actions()
        # Agent instance:
        self.agent = DQNAgent(num_features=self.features.num_features,
                              num_actions=self.actions.get_num_actions(),
                              learning_rate=0.005, discount_factor=0.99,
                              epsilon=1, final_epsilon=0.001,
                              epsilon_decay=0.99995, tau=0.125)
    
    def get_reward(self, game_status: int) -> int:
        if game_status == GOAL:
            return 1000
        elif game_status in [CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME]:
            return -1000
        else:
            return -1
    
    def set_starting_game_conditions(self, game_interface: HFOAttackingPlayer,
                                     features: PlasticFeatures,
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
            try:
                self.game_interface.check_server_is_up()
            except ServerDownError as e:
                print("!!SERVER DOWN!! TEST {}/{}".format(ep, num_episodes))
                avr_win_rate = round(_num_wins / (ep+1), 2)
                print("[TEST: Summary] WIN rate = {};".format(avr_win_rate))
                return avr_win_rate
            # Update features:
            self.features.update_features(self.game_interface.get_state())
            # Set up gaming conditions:
            self.set_starting_game_conditions(
                game_interface=self.game_interface, features=self.features,
                start_pos=starting_pos_list[ep % len(starting_pos_list)],
                start_with_ball=start_with_ball)
            print("\nNEW TEST [{}]".format(
                starting_pos_list[ep % len(starting_pos_list)]))
            # print("FEATURES: ", self.features.get_features())

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
        avr_win_rate = round(_num_wins / num_episodes, 2)
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
        starting_pos_list = list(STARTING_POSITIONS.values())
        
        # metrics variables:
        _num_wins = 0
        _sum_epsilons = 0
        for ep in range(num_train_episodes):
            # Check if server still running:
            try:
                self.game_interface.check_server_is_up()
            except ServerDownError as e:
                print("!!SERVER DOWN!! TRAIN {}/{}".
                      format(ep, num_train_episodes))
                return
            # Update features:
            self.features.update_features(self.game_interface.get_state())
            
            # Go to origin position:
            self.set_starting_game_conditions(
                game_interface=self.game_interface, features=self.features,
                start_pos=starting_pos_list[ep % len(starting_pos_list)],
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
                    # Train:
                    self.agent.train(terminal_state=done)
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
            # Add episodes:
            self.agent.store_episode(episode_buffer, reward=reward)
            # Update auxiliar variables:
            _sum_epsilons += self.agent.epsilon
            # Update Agent:
            self.agent.restart(num_total_train_ep)
            # Game Reset
            self.game_interface.reset()
        avr_epsilon = round(_sum_epsilons / num_train_episodes, 3)
        print("[TRAIN: Summary] WIN rate = {}; AVR epsilon = {}".format(
            _num_wins / num_train_episodes, avr_epsilon))
        return avr_epsilon

    def train_offline(self, game_buffer: np.ndarray):
        for _ in range(5):
            buffer = game_buffer.copy()
            self.agent.train_from_batch(buffer)
            print("MODEL TRAINED")
            aux = [-1] * 6
            features_base = np.array(aux)
            for idx in range(6):
                features_array = features_base.copy()
                features_array[idx] = 0
                print("[TEST] {}".format(features_array.tolist()))
                action_idx = self.agent.exploit_actions(features_array,
                                                        verbose=True)
                print("-> {}".format(
                    self.actions.map_action_to_str(action_idx)))
