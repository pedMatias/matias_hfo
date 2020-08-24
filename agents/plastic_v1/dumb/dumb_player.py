#!/usr/bin/env python3
# encoding utf-8
import random
import json
import os
import argparse

import numpy as np
from hfo import GOAL, IN_GAME, CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS

from agents.dqn_v1.aux import mkdir
from agents.utils import ServerDownError
import settings
from agents.plastic_v0.base.hfo_attacking_player import HFOAttackingPlayer
from agents.plastic_v0.actions.plastic import Actions
from agents.plastic_v0.features.plastic_features import PlasticFeatures

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
        self.features = PlasticFeatures(num_op=num_opponents,
                                        num_team=num_teammates)
        # Actions Interface:
        self.actions = Actions(num_team=num_teammates, features=self.features,
                               game_interface=self.game_interface)

    def get_reward(self, game_status: int) -> int:
        if game_status == GOAL:
            return 1000
        elif game_status in [CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME]:
            return -1000
        else:
            return -1

    def set_starting_game_conditions(self, game_interface: HFOAttackingPlayer,
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
            self.actions.dribble_to_pos(start_pos)
        else:
            if self.features.has_ball():
                self.actions.kick_to_pos((0, 0))
            # Move to starting position:
            self.actions.move_to_pos(start_pos)
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
            except ServerDownError:
                print("!!SERVER DOWN!! TEST {}/{}".format(ep, num_episodes))
                avr_win_rate = round(_num_wins / (ep + 1), 2)
                print("[TEST: Summary] WIN rate = {};".format(avr_win_rate))
                return avr_win_rate
    
            # Update features:
            self.features.update_features(self.game_interface.get_observation())
    
            # Set up gaming conditions:
            self.set_starting_game_conditions(
                game_interface=self.game_interface,
                start_pos=starting_pos_list[ep % len(starting_pos_list)],
                start_with_ball=start_with_ball)
            print("\nNEW TEST [{}]".format(
                starting_pos_list[ep % len(starting_pos_list)]))
    
            # Start learning loop
            status = IN_GAME
            prev_act = None
            while self.game_interface.in_game():
                if self.features.has_ball():
                    # Update environment features:
                    features_array = self.features.get_features().copy()

                    # Act:
                    act = self.agent.exploit_actions(features_array)
                    if prev_act != act:
                        print(f"ACTION:: {self.actions.action_w_ball[act]}")
                    prev_act = act
                    self.actions.execute_action(act, with_ball=True)
                else:
                    if prev_act != -1:
                        print("ACTION:: MOVE!!")
                    prev_act = -1
                    status = self.actions.execute_action(0, with_ball=False)

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
        starting_pos_list = list(STARTING_POSITIONS.values())

        # metrics variables:
        _num_wins = 0
        _sum_epsilons = 0
        for ep in range(num_train_episodes):
            # Check if server still running:
            try:
                self.game_interface.check_server_is_up()
            except ServerDownError:
                print("!!SERVER DOWN!! TRAIN {}/{}".
                      format(ep, num_train_episodes))
                return
            # Update features:
            self.features.update_features(self.game_interface.get_observation())
    
            # Go to origin position:
            self.set_starting_game_conditions(
                game_interface=self.game_interface,
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
                    act = self.fix_action(features_array)
                    status = self.actions.execute_action(act, with_ball=True)
                else:
                    status = self.actions.execute_action(0, with_ball=False)
            if self.game_interface.scored_goal() or status == GOAL:
                _num_wins += 1
                reward = self.get_reward(GOAL)
            else:
                reward = self.get_reward(status)
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
    parser.add_argument('--num_episodes', type=int, default=0)
    parser.add_argument('--starts_with_ball', type=bool, default=True)
    
    # Parse Arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_train_ep = args.num_episodes
    starts_with_ball = args.starts_with_ball
    
    # Start Player:
    player = Player(num_teammates=num_team, num_opponents=num_op)
    
    print("\n[{} - PLAYER] num_opponents={}; num_teammates={}; "
          "start_with_ball={}".format("RETRAIN" if args.retrain else "TRAIN",
                                      num_op, num_op, starts_with_ball))
    
    
    player.train(num_episodes=,
                 start_with_ball=starts_with_ball)
    
    print("\n\n!!!!!!!!! AGENT EXIT !!!!!!!!!!!!\n\n")