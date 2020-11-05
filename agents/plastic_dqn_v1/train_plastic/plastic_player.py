# !/usr/bin/env python3
# encoding utf-8
import argparse
import json
import os
import random
from shutil import copyfile

import numpy as np

import settings
from agents.utils import ServerDownError, get_vertices_around_ball
from agents.plastic_dqn_v1.base.hfo_attacking_player import HFOAttackingPlayer
from agents.plastic_dqn_v1.agent.dqn import DQN
from agents.plastic_dqn_v1.actions.complex import Actions
from agents.plastic_dqn_v1.features.plastic_features import PlasticFeatures
from agents.plastic_dqn_v1.plastic import Policy
from agents.plastic_dqn_v1 import config



"""
This module is used to test the agent previous trained
"""


class GameMetrics:
    def __init__(self):
        self.num_ep = 0
        self.num_wins = 0
        self.num_wrong_actions = 0
        self.num_correct_actions = 0
        self.num_games_touched_ball = 0
        self.num_games_passed_ball = 0
    
    def restart(self):
        self.num_ep = 0
        self.num_wins = 0
        self.num_wrong_actions = 0
        self.num_correct_actions = 0
        self.num_games_touched_ball = 0
        self.num_games_passed_ball = 0
    
    def inc_num_ep(self):
        self.num_ep += 1
    
    def inc_num_wins(self):
        self.num_wins += 1
    
    def inc_num_games_touched_ball(self):
        self.num_games_touched_ball += 1
    
    def inc_num_games_passed_ball(self):
        self.num_games_passed_ball += 1
    
    def inc_num_wrong_actions(self):
        self.num_wrong_actions += 1
    
    def inc_num_correct_actions(self):
        self.num_correct_actions += 1
    
    def export(self, num_episodes: int, verbose: bool = True):
        correct_actions_rate = \
            round(self.num_correct_actions /
                  (self.num_correct_actions + self.num_wrong_actions),
                  2)
        avr_touched_ball_rate = \
            round(self.num_games_touched_ball / num_episodes, 2)
        avr_passed_ball_rate = round(
            self.num_games_passed_ball / max(self.num_games_touched_ball, 1),
            2)
        win_rate = self.num_wins / num_episodes
        if verbose:
            print(f"[Game Metrics: Summary] WIN rate = {win_rate}; "
                  f"Correct Actions Rate = {correct_actions_rate}; "
                  f"Touched Ball rate = {avr_touched_ball_rate}; "
                  f"Pass Ball rate={avr_passed_ball_rate};")
        return win_rate, correct_actions_rate, avr_touched_ball_rate, \
               avr_passed_ball_rate
    
    def export_to_dict(self, num_episodes: int, verbose: bool = True):
        win_rt, corr_act_rt, avr_ball_rt, avr_pass_rt = \
            self.export(num_episodes=num_episodes, verbose=verbose)
        return dict(win_rate=win_rt, correct_actions_rate=corr_act_rt,
                    avr_touched_ball_rate=avr_ball_rt,
                    avr_passed_ball_rate=avr_pass_rt)
    

class Player:
    def __init__(self, num_opponents: int, num_teammates: int, model_file: str,
                 epsilon: int = 1,  port: int = 6000):
        # Game Interface:
        self.game_interface = HFOAttackingPlayer(num_opponents=num_opponents,
                                                 num_teammates=num_teammates,
                                                 port=port)
        self.game_interface.connect_to_server()
        # Features Interface:
        self.features = PlasticFeatures(num_op=num_opponents,
                                        num_team=num_teammates)
        # Actions Interface:
        self.actions = Actions(num_team=num_teammates, features=self.features,
                               game_interface=self.game_interface)
        # Agent instance:
        self.epsilon = epsilon
        self.dqn = DQN.load(load_file=model_file)
        # Metrics:
        self.metrics = GameMetrics()
    
    def set_starting_game_conditions(self, start_pos: tuple = None,
                                     starts_fixed_position: bool = True,
                                     verbose: bool = False):
        """
        Set starting game conditions. Move for initial position, for example
        """
        if starts_fixed_position:
            if not start_pos:
                ball_pos: list = list(self.features.get_ball_coord())
                starting_corners = get_vertices_around_ball(ball_pos)
                start_pos = random.choice(starting_corners)
            self.actions.move_to_pos(start_pos)
            if verbose:
                print(f"[PLAYER: GAME SET UP] Initial pos= {start_pos}")
        else:
            # Start in current position
            if verbose:
                print(f"[START GAME] Initial pos= RANDOM")
        # Informs the other players that it is ready to start:
        self.game_interface.hfo.say(settings.PLAYER_READY_MSG)
    
    def exploit_actions(self, state: np.ndarray, verbose: bool = False) -> int:
        q_predict = self.dqn.predict(state)
        max_list = np.where(q_predict == q_predict.max())
        if len(max_list[0]) > 1:
            action = np.random.choice(max_list[0])
        else:
            action = np.argmax(q_predict)
        if verbose:
            print("Q values {} -> {}".format(q_predict, int(action)))
        return int(action)

    def play(self, num_episodes: int, starts_fixed_position: bool = True):
        """
        @param num_episodes: number of episodes to train in this iteration
        @param starts_fixed_position: bool
        @raise ServerDownError
        @return: Game Metrics
        """
        self.metrics.restart()
        
        for ep in range(num_episodes):
            # Check if server still running:
            try:
                self.game_interface.check_server_is_up()
            except ServerDownError:
                print("!!SERVER DOWN!! Test {}/{}".format(ep, num_episodes))
                metrics_dict = self.metrics.export_to_dict(num_episodes)
                return metrics_dict

            # Update features:
            self.features.update_features(
                observation=self.game_interface.get_observation())

            # Go to origin position:
            self.set_starting_game_conditions(
                starts_fixed_position=starts_fixed_position)

            # metrics:
            touched_ball = False
            passed_ball = False
            while self.game_interface.in_game():
                if self.features.has_ball(): touched_ball = True
                
                # Update environment features:
                features_array = self.features.get_features()
                
                # Act:
                act = self.exploit_actions(features_array, verbose=False)
                status, correct_action, passed_ball_succ = \
                    self.actions.execute_action(act)
    
                # Metrics:
                if passed_ball_succ is True: passed_ball = True
                if correct_action:
                    self.metrics.inc_num_correct_actions()
                else:
                    self.metrics.inc_num_wrong_actions()

            # Update auxiliar variables:
            self.metrics.inc_num_ep()
            if self.game_interface.scored_goal(): self.metrics.inc_num_wins()
            if touched_ball: self.metrics.inc_num_games_touched_ball()
            if passed_ball: self.metrics.inc_num_games_passed_ball()
            
            # Game Reset
            self.game_interface.reset()

        metrics_dict = self.metrics.export_to_dict(num_episodes)
        return metrics_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--team_name', type=str, default=None)
    parser.add_argument('--starts_fixed_position', type=str, default="true")
    parser.add_argument('--dir', type=str)
    parser.add_argument('--port', type=int, default=6000)
    
    # Parse arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_episodes
    team_name = args.team_name
    starts_fixed_position = True if args.starts_fixed_position == "true" \
        else False
    directory = args.dir
    port = args.port

    print(f"[PLASTIC Player: {team_name}] ep={num_episodes}; "
          f"num_t={num_team}; num_op={num_op}; "
          f"starts_fixed_pos={starts_fixed_position};")
    
    # Model file:
    base_model_file = config.MODEL_FILE_FORMAT.format(team_name=team_name,
                                                      step=step)
    prefix = "new_" if new_model else "re_"
    base_model_file = prefix + base_model_file
    model_file = os.path.join(directory, base_model_file + f".{model_idx}")
    player = Player(num_teammates=num_team, num_opponents=num_op,
                    model_file=model_file)

    # Test Player
    game_metrics: dict = player.play(
        num_episodes=num_episodes,
        starts_fixed_position=starts_fixed_position)
        
    metrics_file = os.path.join(directory,
                                f"{team_name}_test_metrics_{step}.json")
    # Load metrics data:
    if os.path.isfile(metrics_file):
        with open(metrics_file, "rb") as fp:
            test_metrics = json.load(fp)
    else:
        test_metrics = dict()
    
    key = "new_models" if new_model else "re_trained_models"
    if test_metrics.get(key):
        test_metrics[key][model_idx] = game_metrics
    else:
        test_metrics[key] = {model_idx: game_metrics}
    with open(metrics_file, 'w+') as fp:
        json.dump(test_metrics, fp)

    print("\n!!!!!!!!! Test End !!!!!!!!!!!!\n\n")
