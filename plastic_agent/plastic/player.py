# !/usr/bin/hfo_env python3
# encoding utf-8
import os

from hfo import GOAL, CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS

from plastic_agent import config
from agents.utils import ServerDownError
from plastic_agent.agent.replay_buffer import Transition
from plastic_agent.plastic.policy import Policy
from plastic_agent.plastic.behaviour_dist import BehaviourDist
from plastic_agent.plastic.metrics import EpisodeMetrics, GameMetrics
from plastic_agent.hfo_env.game_interface import GameInterface
from plastic_agent.hfo_env.features.plastic import PlasticFeatures
from plastic_agent.hfo_env.actions.plastic import PlasticActions

"""
This module is used to test the agent previous trained
"""


class PlasticPlayer:
    def __init__(self, team_name: str, num_opponents: int, num_teammates: int,
                 models_dir: str, memory_bounded: bool = False,
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
            history_len=history_len
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
    
    def _get_reward(self, game_status: int, correct_action: bool) -> int:
        reward = 0
        if game_status == GOAL:
            reward += 1000
        elif game_status in [CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME]:
            reward -= 1000
        else:
            if correct_action:
                reward += 1
            else:
                reward -= 5
        return reward
    
    def _play_episode(self, verbose: bool = False):
        # auxiliar structures:
        guessed_teams = list()
        # metrics:
        touched_ball = False
        passed_ball = False
        num_wrong_actions = 0
        num_correct_actions = 0
        while self.game_interface.in_game():
            if self.features.has_ball():
                touched_ball = True
            # Update environment features:
            features_array = self.features.get_features()
            # Act:
            act = self.behaviour_dist.select_action(features_array)
            status, correct_action, passed_ball_succ = \
                self.actions.execute_action(act, verbose=verbose)

            # Store transition:
            # (obs, action, reward, new obs, done?)
            transition = Transition(
                obs=features_array,
                act=act,
                reward=self._get_reward(status, correct_action),
                new_obs=self.features.get_features(),
                done=not self.game_interface.in_game(),
                correct_action=correct_action
            )

            self.behaviour_dist.update_beliefs(transition, verbose=verbose)
            predicted_policy = self.behaviour_dist.get_best_policy()
            guessed_teams.append(predicted_policy.team_name)
            
            # Metrics:
            if passed_ball_succ:
                passed_ball = True
            if correct_action:
                num_correct_actions += 1
            else:
                num_wrong_actions += 1
        metrics = EpisodeMetrics(
            touched_ball=touched_ball,
            passed_ball=passed_ball,
            num_wrong_actions=num_wrong_actions,
            num_correct_actions=num_correct_actions)
        return guessed_teams, metrics
    
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
        
        for ep in range(num_episodes):
            # Check if server still running:
            try:
                self.game_interface.check_server_is_up()
            except ServerDownError:
                print("!!SERVER DOWN!! Test {}/{}".format(ep, num_episodes))
                return selected_teams, game_metrics.export_to_dict()
            
            # Update features:
            self.features.update_features(
                observation=self.game_interface.get_observation())
            # Play episode:
            guessed_teams, ep_metrics = self._play_episode(verbose=verbose)
            if verbose:
                _perct = {}
                _steps = len(guessed_teams)
                for aux_name in config.TEAMS_NAMES:
                    _perct[aux_name] = guessed_teams.count(aux_name) / _steps
                print(f"[{ep}: Teams predicted] {_perct};")
            
            # Update auxiliar variables:
            game_metrics.add_episode_metrics(
                ep_metrics,
                goal=self.game_interface.scored_goal(),
                guessed_teams=guessed_teams
            )
            
            # Selected Teams:
            aux = {}
            steps = len(guessed_teams)
            for team_name in self.behaviour_dist.team_names:
                if team_name in guessed_teams:
                    aux[team_name] = guessed_teams.count(team_name) / steps
                else:
                    aux[team_name] = 0
            selected_teams.append(aux)
            
            # Game Reset
            self.game_interface.reset()
        
        metrics_dict = game_metrics.export_to_dict()
        metrics_dict["teams"] = [policy.team_name for policy in self.policies]
        metrics_dict["correct_team"] = self.team_name
        if verbose:
            print(f"[Game Metrics] {metrics_dict}")
        return selected_teams, metrics_dict
