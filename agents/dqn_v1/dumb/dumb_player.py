#!/usr/bin/hfo_env python3
# encoding utf-8
import random
import json
import os
import argparse

import numpy as np
from hfo import GOAL, IN_GAME

from agents.dqn_v1.aux import mkdir
from agents.utils import ServerDownError
import settings
from agents.base.hfo_attacking_player import HFOAttackingPlayer
from agents.dqn_v1.q_agent import QAgent
from environement_features.reward_functions import basic_reward
from agents.dqn_v1.actions.simple import Actions
from agents.dqn_v1.features.discrete_features import DiscFeatures1Teammate

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
    
    def fix_action(self, features_array: np.ndarray) -> int:
        """ ["KICK_TO_GOAL", "DRIBBLE_UP", "DRIBBLE_DOWN", "DRIBBLE_LEFT",
        "DRIBBLE_RIGHT"]"""
        # TOP LEFT:
        if features_array[0] == 1:
            return 4
        # TOP RIGHT
        elif features_array[1] == 1:
            return 2
        # MID LEFT
        elif features_array[2] == 1:
            return 4
        # MID RIGHT
        elif features_array[3] == 1:
            return 0
        # BOTTOM LEFT
        elif features_array[4] == 1:
            return 4
        # BOTTOM RIGHT
        elif features_array[5] == 1:
            return 1

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
                    ## action_idx = self.agent.act(features_array)
                    action_idx = self.fix_action(features_array)
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


def export_metrics(trained_eps: list, avr_win_rate: list, epsilons: list,
                   save_dir: str):
    """ Saves metrics in Json file"""
    data = {"trained_eps": trained_eps, "epsilons": epsilons,
            "avr_win_rate": avr_win_rate}
    file_path = os.path.join(save_dir, "metrics.json")
    with open(file_path, 'w+') as fp:
        json.dump(data, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_train_ep', type=int, default=1000)
    parser.add_argument('--num_test_ep', type=int, default=0)
    parser.add_argument('--num_repetitions', type=int, default=0)
    parser.add_argument('--retrain', type=bool, default=False)
    parser.add_argument('--starts_with_ball', type=bool, default=True)
    parser.add_argument('--load_file', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    # Parse Arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_train_ep = args.num_train_ep
    num_test_ep = args.num_test_ep
    num_repetitions = args.num_repetitions
    num_episodes = (num_train_ep + num_test_ep) * num_repetitions
    starts_with_ball = args.starts_with_ball
    
    # Start Player:
    player = Player(num_teammates=num_team, num_opponents=num_op)
    
    # IF retrain mode, load previous model
    if args.retrain and args.load_file:
        player.agent.load_model(args.load_file)
    
    # Directory
    save_dir = args.save_dir or mkdir(
        num_episodes, num_op,
        extra_note="retrain" if args.retrain else "new")
    
    print("\n[{} - PLAYER] num_opponents={}; num_teammates={}; "
          "start_with_ball={}".format("RETRAIN" if args.retrain else "TRAIN",
                                      num_op, num_op, starts_with_ball))
    
    # Test one first time without previous train:
    ##av_reward = player.test(num_episodes=num_test_ep,
    ##                        start_with_ball=starts_with_ball)
    # Save metrics structures
    trained_eps_list = [0]
    avr_epsilons_list = [player.agent.epsilon]
    avr_win_rate = [0]  # av_reward]
    
    # Train - test iterations:
    for i in range(num_repetitions):
        print(">>>> {}/{} <<<<".format(i, num_repetitions))
        try:
            # Train:
            player.train(num_train_episodes=num_train_ep,
                         num_total_train_ep=num_train_ep * num_repetitions,
                         start_with_ball=starts_with_ball)
            # Test:
            av_reward = player.test(num_episodes=num_test_ep,
                                    start_with_ball=starts_with_ball)
        except ServerDownError as e:
            print("\n!!! Server is Down !!!")
            pass
            av_reward = 0
        sum_trained_eps = trained_eps_list[-1] + num_train_ep
        # Calc metrics:
        trained_eps_list.append(sum_trained_eps)
        avr_epsilons_list.append(player.agent.epsilon)
        avr_win_rate.append(av_reward)
    print("\n\n!!!!!!!!! AGENT FINISHED !!!!!!!!!!!!\n\n")
    # Save and export metrics:
    player.agent.save_model(file_name=save_dir + "/agent_model")
    export_metrics(trained_eps=trained_eps_list, avr_win_rate=avr_win_rate,
                   epsilons=avr_epsilons_list, save_dir=save_dir)
    print("\n\n!!!!!!!!! AGENT EXIT !!!!!!!!!!!!\n\n")