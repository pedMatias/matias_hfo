# !/usr/bin/hfo_env python3
# encoding utf-8
import numpy as np

from multi_agents.dqn_agent.player import Player
from multi_agents.dqn_agent.replay_buffer import Transition
from multi_agents.dqn_agent.metrics import GameMetrics, EpisodeMetrics

"""
This module is used to test the agent previous trained
"""


class FixedPlayer(Player):
    def __init__(self, team_name: str, num_opponents: int, num_teammates: int,
                 model_file: str, epsilon: int = 1, port: int = 6000,
                 learning_boost: bool = False, pass_ball: bool = False):
        print("!!!! Fixed Player !!!!")
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
            a_goal_angle = self.features.agent.goal_opening_angle
            
            # Pass ball:
            if self.pass_ball and np.random.random() < 0.3:
                return aux_pass(a_goal_angle)
            # Shoot:
            if self.features.agent.dist_to_goal <= -0.4:
                # Shoot:
                if a_goal_angle > -0.747394386098:
                    return 7  # Shoot
                # Pass Ball:
                elif self.pass_ball:
                    return aux_pass(a_goal_angle)
                else:
                    if self.features.near_opponent():
                        return 7
                    else:
                        return 8 # Short Dribble
            # SHORT_DRIBBLE
            elif -0.1 > self.features.agent.dist_to_goal > -0.4:
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
    
    def act(self, state: np.ndarray, metrics: GameMetrics,
            verbose: bool = False, fixed_mode: bool = False):
        if fixed_mode:
            if verbose: print("[ACT] Fixed Behaviour")
            return self.fixed_behaviour()
        elif np.random.random() < self.epsilon:  # Explore
            if verbose: print("[ACT] Explored")
            metrics.inc_num_exploration_steps()
            return self.explore_actions()
        else:  # Exploit
            if verbose: print("[ACT] Exploit")
            metrics.inc_num_exploitation_steps()
            return self.exploit_actions(state)
    
    def play_episode(self, game_metrics: GameMetrics):
        # auxiliar structures:
        episode_buffer = list()
        # metrics:
        touched_ball = False
        passed_ball = False
        scored_goal = False
        # Fixed play mode?
        if self.learning_boost and np.random.random() < self.epsilon:
            learning_boost = True
            print("[ACT] Fixed Behaviour")
        else:
            learning_boost = False
        while self.game_interface.in_game():
            if self.features.has_ball():
                touched_ball = True
            # Update environment features:
            features_array = self.features.get_features()
            # Act:
            act = self.act(features_array, metrics=game_metrics,
                           fixed_mode=learning_boost)
            self.actions.execute_action(act, verbose=False)

            # Store transition:
            # (obs, action, reward, new obs, done?)
            transition = Transition(
                obs=features_array,
                act=act,
                reward=self.get_reward(self.game_interface.get_game_status()),
                new_obs=self.features.get_features(),
                done=not self.game_interface.in_game()
            )
            episode_buffer.append(transition)
            
            # Metrics:
            if "PASS" in self.actions.get_action_name(action_idx=act):
                passed_ball = True
                
        if self.game_interface.scored_goal():
            uniform = self.game_interface.hfo.getUnum()
            if self.game_interface.last_player_to_touch_ball == uniform:
                scored_goal = True

        metrics = EpisodeMetrics(
            touched_ball=touched_ball,
            passed_ball=passed_ball,
            scored_goal=scored_goal)
        return episode_buffer, metrics
