# !/usr/bin/hfo_env python3
# encoding utf-8
import numpy as np

from multi_agents.dqn_agent.player import Player
from multi_agents.dqn_agent.replay_buffer import Transition
from multi_agents.dqn_agent.metrics import GameMetrics, EpisodeMetrics

"""
This module is used to test the agent previous trained
"""


class TrashPlayer(Player):
    def __init__(self, team_name: str, num_opponents: int, num_teammates: int,
                 model_file: str, epsilon: int = 1, port: int = 6000,
                 learning_boost: bool = False, pass_ball: bool = False):
        print("!!!! TRASH Player !!!!")
        # Game Interface:
        super().__init__(team_name, num_opponents, num_teammates, model_file,
                         epsilon, port)
        # Fixed behviour presented to the agent to help him learn:
        self.learning_boost = learning_boost
        self.pass_ball = pass_ball
    
    def fixed_behaviour(self):
        def aux_pass(a_goal_angle):
            for t_idx, t_player in enumerate(self.features.teammates):
                if (t_player.goal_angle > a_goal_angle) and \
                        (t_player.proximity_op > -0.9) and \
                        (t_player.pass_angle > 0):
                    return 10 + t_idx
                else:
                    return np.random.choice([7, 10])
            
        goal_coord = np.array([0.8, 0])
        if self.features.has_ball():
            dist_to_goal = abs(np.linalg.norm(goal_coord -
                                              self.features.agent_coord))
            a_goal_angle = self.features.agent.goal_opening_angle
            
            # Shoot:
            if abs(dist_to_goal) <= 0.4:
                print(f"NEAR GOAL: {self.features.features_vect[3:6]}")
                # Shoot:
                if a_goal_angle > -0.747394386098:
                    return 7  # Shoot
                # Pass Ball:
                elif self.pass_ball:
                    return aux_pass(a_goal_angle)
                else:
                    return 7
            # SHORT_DRIBBLE
            elif 0.7 > abs(dist_to_goal) > 0.4:
                if self.features.near_opponent():
                    return aux_pass(a_goal_angle)
                else:
                    return 8
            # LONG_DRIBBLE
            else:
                if self.features.near_opponent():
                    return aux_pass(a_goal_angle)
                else:
                    return 9
        else:
            # Move to Goal:
            if self.features.teammates_have_ball():
                return 2
            # MOVE_TO_BALL
            else:
                return 1
    
    def act(self, a1=None, a2=None, a3=None):
            return self.fixed_behaviour()
    
    def play_episode(self, game_metrics: GameMetrics):
        while self.game_interface.in_game():
            if self.features.has_ball():
                touched_ball = True
            # Update environment features:
            features_array = self.features.get_features()
            # Act:
            act = self.act()
            self.actions.execute_action(act, verbose=False)
        return
