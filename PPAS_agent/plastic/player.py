# !/usr/bin/hfo_env python3
# encoding utf-8
import os

from hfo import GOAL, CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS

from multi_agents import config
from agents.utils import ServerDownError
from multi_agents.dqn_agent.replay_buffer import Transition
from multi_agents.plastic.policy import Policy
from multi_agents.plastic.behaviour_dist import BehaviourDist
from multi_agents.plastic.metrics import EpisodeMetrics, GameMetrics
from multi_agents.hfo_env.game_interface import GameInterface
from multi_agents.hfo_env.features.plastic import PlasticFeatures
from multi_agents.hfo_env.actions.plastic import PlasticActions

"""
This module is used to test the agent previous trained
"""


class PlasticPlayer:
    def __init__(self, team_name: str, num_opponents: int, num_teammates: int,
                 models_dir: str, model_type: str, memory_bounded:bool = False,
                 history_len: int = 1, port: int = 6000):
        self.team_name = team_name
        # Game Interface:
        self.game_interface = GameInterface(
            team_name=team_name,
            num_opponents=num_opponents,
            num_teammates=num_teammates,
            port=port)
        # Features Interface:
        self.features = PlasticFeatures(num_op=num_opponents,
                                        num_team=num_teammates)
        # Actions Interface:
        self.actions = PlasticActions(num_team=num_teammates,
                                      features=self.features,
                                      game_interface=self.game_interface)
        # Agent instance:
        self.policies = self.load_plastic_policies(models_dir,
                                                   config.TEAMS_NAMES)
        self.behaviour_dist = BehaviourDist(
            policies=self.policies,
            memory_bounded=memory_bounded,
            history_len=history_len,
            num_features=self.features.get_num_features(),
            model_type=model_type
        )
        # Connect to rccserver
        self.game_interface.connect_to_server()
    
    @staticmethod
    def load_plastic_policies(dir_path: str, team_names: list):
        if not os.path.isdir(dir_path):
            print(f"[load_plastic_models] Dir not found {dir_path};")
            raise NotADirectoryError(dir_path)
        policies = list()
        for team_name in team_names:
            if not os.path.isdir(os.path.join(dir_path, team_name)):
                print(f":: Can not find team {team_name}!\n".upper())
            else:
                policy = Policy.load(team_name=team_name, base_dir=dir_path)
                policies.append(policy)
                print(f":: Found Policy {team_name};")
        return policies
    
    def _get_reward(self, game_status: int) -> int:
        reward = 0
        if game_status == GOAL:
            kicker_unum = self.game_interface.get_last_player_to_touch_ball()
            # Player scored the goal:
            if kicker_unum == 11:
                reward += 100
            reward += 1000
        elif game_status in [CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME]:
            reward -= 1000
        else:
            reward -= 1
        return reward
    
    def _play_episode(self, verbose: bool = False):
        # auxiliar structures:
        guessed_teams = list()
        b_dist_buffer = list()
        # metrics:
        touched_ball = False
        passed_ball = False
        while self.game_interface.in_game():
            if self.features.has_ball():
                touched_ball = True
            # Update environment features:
            features_array = self.features.get_features()
            # Act:
            legal_actions = self.actions.get_legal_actions()
            act = self.behaviour_dist.select_action(features_array,
                                                    legal_actions)
            self.actions.execute_action(act, verbose=verbose)

            # Store transition:
            # (obs, action, reward, new obs, done?)
            transition = Transition(
                obs=features_array,
                act=act,
                reward=self._get_reward(self.game_interface.get_game_status()),
                new_obs=self.features.get_features(),
                done=not self.game_interface.in_game()
            )
            
            # Update Beliefs:
            self.behaviour_dist.update_beliefs(transition)
            
            # Save metrics:
            predicted_policy = self.behaviour_dist.get_best_policy()
            guessed_teams.append(predicted_policy.team_name)
            b_dist_buffer.append(self.behaviour_dist.get_probabilities_dict())
            
            # Metrics:
            if "PASS" in self.actions.get_action_name(action_idx=act):
                passed_ball = True
        metrics = EpisodeMetrics(
            touched_ball=touched_ball,
            passed_ball=passed_ball)
        return guessed_teams, b_dist_buffer, metrics
    
    def play(self, num_episodes: int, verbose: bool = False):
        """
        @param num_episodes: number of episodes to train in this iteration
        @raise ServerDownError
        @return: Selected Teams, Game Metrics
        """
        game_metrics = GameMetrics()
        game_metrics.set_correct_team(self.team_name)
        # Predicted Teams Distributions
        selected_teams = list()
        game_results = list()
        
        for ep in range(num_episodes):
            # Check if server still running:
            try:
                self.game_interface.check_server_is_up()
            except ServerDownError:
                print("!!SERVER DOWN!! Test {}/{}".format(ep, num_episodes))
                return selected_teams, game_results, \
                       game_metrics.export_to_dict()
            
            # Update features:
            self.features.re_calculate_features(
                observation=self.game_interface.get_observation(),
                last_player_touch_ball_uniform_num=0)
            # Play episode:
            guessed_teams, b_dist_buffer, ep_metrics = \
                self._play_episode(verbose=verbose)
            goal: bool = self.game_interface.scored_goal()
            
            # Update auxiliar variables:
            game_metrics.add_episode_metrics(
                ep_metrics,
                goal=goal,
                guessed_teams=guessed_teams
            )
            
            game_results.append(1 if goal else 0)
            # Selected Teams:
            aux_dict = dict()
            for ep_dist in b_dist_buffer:
                for team, val in ep_dist.items():
                    try:
                        aux_dict[team] += val
                    except KeyError:
                        aux_dict[team] = val
            # Normalize values:
            num_ep = len(b_dist_buffer)
            for team, val in aux_dict.items():
                aux_dict[team] = val / num_ep
            selected_teams.append(aux_dict)
            
            # Game Reset
            self.game_interface.reset()
        
        metrics_dict = game_metrics.export_to_dict()
        metrics_dict["teams"] = [policy.team_name for policy in self.policies]
        metrics_dict["correct_team"] = self.team_name
        if verbose:
            print(f"[Game Metrics] {metrics_dict}")
        return selected_teams, game_results, metrics_dict
