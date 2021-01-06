#!/usr/bin/hfo_env python3
# encoding utf-8
import random

import settings
from agents.base.hfo_attacking_player import HFOAttackingPlayer
from agents.agent_module_dqn.deep_agent import DQNAgent
from environement_features.reward_functions import basic_reward
from actions_levels.action_module import DiscreteActionsModule
from agents.agent_module_dqn.features.discrete_features import \
    DiscreteFeatures1Teammate

STARTING_POSITIONS = {"TOP LEFT": (-0.5, -0.7), "TOP RIGHT": (0.4, -0.7),
                      "MID LEFT": (-0.5, 0.0), "MID RIGHT": (0.4, 0.0),
                      "BOTTOM LEFT": (-0.5, 0.7), "BOTTOM RIGHT": (0.4, 0.7)}


class Player:
    def __init__(self, num_opponents: int, num_teammates: int):
        # Game Interface:
        self.game_interface = HFOAttackingPlayer(num_opponents=num_opponents,
                                                 num_teammates=num_teammates)
        self.game_interface.connect_to_server()
        # Features Interface:
        self.features = DiscreteFeatures1Teammate(num_op=num_opponents,
                                                  num_team=num_teammates)
        # Actions Interface:
        self.actions = DiscreteActionsModule()
        # Agent instance:
        self.agent = DQNAgent(num_features=self.features.num_features,
                              num_actions=self.actions.get_num_actions(),
                              learning_rate=0.1, discount_factor=0.9,
                              epsilon=0.8)
    
    def get_reward(self, status: int) -> int:
        return basic_reward(status)
    
    def set_starting_game_conditions(self, game_interface: HFOAttackingPlayer,
                                     features: DiscreteFeatures1Teammate,
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

    def test(self, num_episodes: int, start_with_ball:bool = True) -> float:
        """
        @param num_episodes: number of episodes to run
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

            # Start learning loop
            prev_action_idx = None
            while self.game_interface.in_game():
                # Update environment features:
                features_array = self.features.get_features().copy()
    
                # Act:
                action_idx = self.agent.exploit_actions(features_array)
                if prev_action_idx != action_idx:
                    print("ACTION:: {}".format(
                        self.actions.map_action_to_str(
                            action_idx, self.features.has_ball())))
                prev_action_idx = action_idx
                
                self.actions.execute_action(
                    action_idx=action_idx,
                    features=self.features,
                    game_interface=self.game_interface)

            # Update auxiliar variables:
            _num_wins += 1 if self.game_interface.scored_goal() else 0
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
            while self.game_interface.in_game():
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
                self.agent.store_transition(
                    curr_st=features_array,
                    action_idx=action_idx,
                    reward=self.get_reward(status),
                    new_st=self.features.get_features(),
                    done=done)
                self.agent.train(done)
            
            # Update auxiliar variables:
            _sum_epsilons += self.agent.epsilon
            _num_wins += 1 if self.game_interface.scored_goal() else 0
            # Update Agent:
            self.agent.restart(num_total_train_ep)
            # Game Reset
            self.game_interface.reset()
        print("[TRAIN: Summary] WIN rate = {}; AVR epsilon = {}".format(
            _num_wins / num_train_episodes, _sum_epsilons / num_train_episodes))
