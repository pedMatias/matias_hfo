# !/usr/bin/env python3
# encoding utf-8
import argparse
import json
import os
import random
from shutil import copyfile

import settings
from agents.utils import ServerDownError, get_vertices_around_ball
from agents.offline_plastic_v2.base.hfo_attacking_player import HFOAttackingPlayer
from agents.offline_plastic_v2.deep_agent import DQNAgent
from agents.offline_plastic_v2.actions.complex import Actions
from agents.offline_plastic_v2.features.plastic_features import PlasticFeatures


class Player:
    def __init__(self, num_opponents: int, num_teammates: int,
                 port: int = 6000):
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
        self.agent = DQNAgent(num_features=self.features.num_features,
                              num_actions=self.actions.get_num_actions(),
                              create_model=False)
        # Auxiliar attributes
        self.num_ep = 0
    
    def test(self, num_episodes: int, verbose: bool = True) -> \
            (float, float, float):
        """
        @param num_episodes: number of episodes to run
        @param start_with_ball: flag
        @param starts_fixed_position: flag
        @return: (win rate, correct actions rate, touched_ball)
        """
        # metrics variables:
        _num_wins = 0
        _num_wrong_actions = 0
        _num_correct_actions = 0
        _num_games_touched_ball = 0
        _num_games_passed_ball = 0
        self.num_ep = 0
        for ep in range(num_episodes):
            # Check if server still running:
            try:
                self.game_interface.check_server_is_up()
            except ServerDownError:
                print("!!SERVER DOWN!! TEST {}/{}".format(ep, num_episodes))
                avr_win_rate = round(_num_wins / (ep + 1), 2)
                print("[TEST: Summary] WIN rate = {};".format(avr_win_rate))
                correct_actions_rate = \
                    round(_num_correct_actions /
                          (_num_correct_actions + _num_wrong_actions),
                          2)
                avr_touched_ball_rate = \
                    round(_num_games_touched_ball / num_episodes, 2)
                avr_passed_ball_rate = \
                    round(_num_games_passed_ball / max(_num_games_touched_ball,
                                                       1), 2)
                return avr_win_rate, correct_actions_rate, \
                    avr_touched_ball_rate, avr_passed_ball_rate
            
            # Update features:
            self.features.update_features(self.game_interface.get_observation())
            
            # Go to origin position:
            print(f"\n[TEST] {ep}/{num_episodes}")
            
            # Start learning loop
            touched_ball = False
            passed_ball = False
            while self.game_interface.in_game():
                # Metrics: Touched ball:
                if self.features.has_ball():
                    touched_ball = True
                    
                # Update environment features:
                features_array = self.features.get_features().copy()
                # Act:
                act = self.agent.exploit_actions(features_array)

                _, act_succ, passed_succ = self.actions.execute_action(
                    act, verbose=verbose)
                
                # Save metric pass:
                if passed_succ is True:
                    passed_ball = True
                
                # Metrics: Number of correct actions::
                if act_succ:
                    _num_correct_actions += 1
                else:
                    _num_wrong_actions += 1
                
            # Update auxiliar variables:
            if touched_ball:
                _num_games_touched_ball += 1
            if passed_ball:
                _num_games_passed_ball += 1
            if self.game_interface.scored_goal():
                print("[WIN] Score GOAL!")
                _num_wins += 1
            else:
                status = self.game_interface.status
                print(f"[LOSS] "
                      f"{self.game_interface.hfo.statusToString(status)}!")
            # Game Reset
            self.game_interface.reset()
            self.num_ep += 1
        avr_win_rate = round(_num_wins / num_episodes, 2)
        correct_actions_rate = \
            round(_num_correct_actions /
                  (_num_correct_actions + _num_wrong_actions),
                  2)
        avr_touched_ball_rate = \
            round(_num_games_touched_ball / num_episodes, 2)
        avr_passed_ball_rate = \
            round(_num_games_passed_ball / max(_num_games_touched_ball, 1), 2)
        return avr_win_rate, correct_actions_rate, avr_touched_ball_rate, \
               avr_passed_ball_rate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--starts_with_ball', type=str, default="true")
    parser.add_argument('--starts_fixed_position', type=str, default="true")
    parser.add_argument('--choose_best_sub_model', type=str, default="false")
    parser.add_argument('--test_iter', type=int, default=0)
    parser.add_argument('--save', type=str, default="true")
    parser.add_argument('--dir', type=str)
    parser.add_argument('--stage')
    parser.add_argument('--port', type=int, default=6000)
    
    # Parse arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_episodes
    directory = args.dir
    # Stage:
    if isinstance(args.stage, str):
        stage = float(args.stage) if '.' in args.stage else int(args.stage)
    else:
        stage = args.stage
    # Multiple tests:
    test_iter = args.test_iter
    save = True if args.save == "true" else False
    choose_best_sub_model = True if args.choose_best_sub_model == "true" \
        else False
    # Game mode:
    starts_with_ball = True if args.starts_with_ball == "true" \
        else False
    starts_fixed_position = True if args.starts_fixed_position == "true" \
        else False
    port = args.port
    
    if choose_best_sub_model:
        # Get model file:
        model_file = os.path.join(directory,
                                  f"agent_model_{stage}.{test_iter}")
    
        # Load Model:
        player = Player(num_teammates=num_team, num_opponents=num_op)
        player.agent.load_model(model_file)
    
        # Test Player
        avr_win_rate, correct_actions_rate, avr_touched_ball, \
        avr_passed_ball_rate = player.test(num_episodes, verbose=False)
    
        # Load Metrics:
        test_metrics_file = os.path.join(directory, "test_metrics.json")
        if stage > 1 and os.path.isfile(test_metrics_file):
            with open(test_metrics_file, "rb") as fp:
                test_metrics = json.load(fp)
        else:
            test_metrics = dict()
            
        print(f"[TEST: Sub_Stage {stage}.{test_iter}] "
              f"WIN rate = {avr_win_rate}; "
              f"Correct actions rate = {correct_actions_rate}; "
              f"Touched ball rate = {avr_touched_ball}; "
              f"Passed ball rate = {avr_passed_ball_rate}")
        
        # Load previous results:
        stage_model_file = os.path.join(directory, f"agent_model_{stage}")
        prev_metrics = test_metrics.get(f"stage_{stage}")
        if not os.path.isfile(stage_model_file) or prev_metrics is None:
            test_metrics[f"stage_{stage}"] = {
                "win_rate": avr_win_rate,
                "best_model": test_iter,
                "correct_actions_avr": correct_actions_rate,
                "touched_ball_rate": avr_touched_ball,
                "avr_passed_ball_rate": avr_passed_ball_rate
            }
            # Copy Stage model:
            try:
                os.remove(stage_model_file)
            except FileNotFoundError:
                pass
            copyfile(model_file, stage_model_file)
            # os.remove(model_file)
        elif avr_win_rate > prev_metrics["win_rate"] \
                and correct_actions_rate >= min(0.8, prev_metrics["correct_actions_avr"]) \
                and avr_touched_ball >= min(0.5, prev_metrics["touched_ball_rate"]):
            test_metrics[f"stage_{stage}"] = {
                "win_rate": avr_win_rate,
                "best_model": test_iter,
                "correct_actions_avr": correct_actions_rate,
                "touched_ball_rate": avr_touched_ball,
                "avr_passed_ball_rate": avr_passed_ball_rate,
                "reason": "greater avr win rate"
            }
            # Copy Stage model:
            try:
                os.remove(stage_model_file)
            except FileNotFoundError:
                pass
            copyfile(model_file, stage_model_file)
            # os.remove(model_file)
        elif avr_passed_ball_rate > (prev_metrics["avr_passed_ball_rate"]+0.1)\
                and avr_win_rate >= min(0.4, (prev_metrics["win_rate"] - 0.1))\
                and correct_actions_rate >= min(0.8, prev_metrics["correct_actions_avr"]) \
                and avr_touched_ball >= min(0.5, prev_metrics["touched_ball_rate"]):
            test_metrics[f"stage_{stage}"] = {
                "win_rate": avr_win_rate,
                "best_model": test_iter,
                "correct_actions_avr": correct_actions_rate,
                "touched_ball_rate": avr_touched_ball,
                "avr_passed_ball_rate": avr_passed_ball_rate,
                "reason": "more balls passed"
            }
            # Copy Stage model:
            try:
                os.remove(stage_model_file)
            except FileNotFoundError:
                pass
            copyfile(model_file, stage_model_file)
            # os.remove(model_file)
        # More correct actions selected:
        elif correct_actions_rate > (prev_metrics["correct_actions_avr"]+0.1) \
                and avr_win_rate >= max((prev_metrics["win_rate"] - 0.1), 0.4)\
                and avr_passed_ball_rate >= min(prev_metrics["avr_passed_ball_rate"], 0.3) \
                and avr_touched_ball >= min(0.5, prev_metrics["touched_ball_rate"]):
            test_metrics[f"stage_{stage}"] = {
                "win_rate": avr_win_rate,
                "best_model": test_iter,
                "correct_actions_avr": correct_actions_rate,
                "touched_ball_rate": avr_touched_ball,
                "avr_passed_ball_rate": avr_passed_ball_rate,
                "reason": "more correct actions choosen"
            }
            # Copy Stage model:
            try:
                os.remove(stage_model_file)
            except FileNotFoundError:
                pass
            copyfile(model_file, stage_model_file)
            # os.remove(model_file)
        else:
            # os.remove(model_file)
            pass

        with open(test_metrics_file, 'w+') as fp:
            json.dump(test_metrics, fp)
        
    else:
        # Get model file:
        model_file = os.path.join(directory, f"agent_model_{stage}")
        print(f"[TEST] Model={model_file}")
    
        # Load Model:
        player = Player(num_teammates=num_team, num_opponents=num_op)
        player.agent.load_model(model_file)

        # Test Player
        avr_win_rate, correct_actions_rate, avr_touched_ball, \
        avr_passed_ball_rate = player.test(num_episodes)

        print(f"[TEST: Sub_Stage {stage}] "
              f"WIN rate = {avr_win_rate}; "
              f"Correct actions rate = {correct_actions_rate}; "
              f"Touched ball rate = {avr_touched_ball}; "
              f"Passed ball rate = {avr_passed_ball_rate}")
        
        if save:
            # Load Metrics:
            test_metrics_file = os.path.join(directory, "test_metrics.json")
            if stage > 1:
                with open(test_metrics_file, "rb") as fp:
                    test_metrics = json.load(fp)
            else:
                test_metrics = dict()
        
            # Test metrics data:
            test_metrics_file = os.path.join(directory, "test_metrics.json")
            if stage > 1:
                with open(test_metrics_file, "rb") as fp:
                    test_metrics = json.load(fp)
            else:
                test_metrics = dict()
        
            # Write train metrics:
            test_metrics[f"stage_{stage}"] = {
                "win_rate": avr_win_rate,
                "correct_actions_avr": correct_actions_rate,
                "avr_touched_ball": avr_touched_ball,
                "avr_passed_ball_rate": avr_passed_ball_rate}

            with open(test_metrics_file, 'w+') as fp:
                json.dump(test_metrics, fp)

    print("\n!!!!!!!!! Test End !!!!!!!!!!!!\n\n")
