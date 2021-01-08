# !/usr/bin/hfo_env python3
# encoding utf-8
import sys

from hfo import GOAL, CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS
import numpy as np

from agents.utils import ServerDownError
from plastic_policy_agent.dqn_agent.dqn import DQN
from plastic_policy_agent.dqn_agent.replay_buffer import LearnBuffer, Transition
from plastic_policy_agent.dqn_agent.metrics import GameMetrics, EpisodeMetrics
from plastic_policy_agent.hfo_env.game_interface import GameInterface
from plastic_policy_agent.hfo_env.features.plastic import PlasticFeatures
from plastic_policy_agent.hfo_env.actions.plastic import PlasticActions

"""
This module is used to test the agent previous trained
"""


class Player:
    def __init__(self, team_name: str, num_opponents: int, num_teammates: int,
                 model_file: str, epsilon: int = 1, port: int = 6000):
        # Game Interface:
        self.game_interface = GameInterface(
            team_name=team_name,
            num_opponents=num_opponents,
            num_teammates=num_teammates,
            port=port)
        self.game_interface.connect_to_server()
        # Features Interface:
        self.features = PlasticFeatures(num_op=num_opponents,
                                        num_team=num_teammates)
        # Actions Interface:
        self.actions = PlasticActions(num_team=num_teammates,
                                      features=self.features,
                                      game_interface=self.game_interface)
        # Agent instance:
        self.epsilon = epsilon
        self.dqn = DQN.load(load_file=model_file)
    
    def exploit_actions(self, state: np.ndarray, verbose: bool = False) -> int:
        q_predict = self.dqn.predict(state)[0]
    
        # Set illegal actions to zero:
        legal_actions = self.actions.get_legal_actions()
        for i in range(len(q_predict)):
            if i not in legal_actions:
                q_predict[i] = -2000
        # Greedy choice:
        max_list = np.where(q_predict == q_predict.max())
        if len(max_list[0]) > 1:
            action = np.random.choice(max_list[0])
        else:
            action = np.argmax(q_predict)
        if verbose:
            print("Q values {} -> {}".format(q_predict, int(action)))
        return int(action)
    
    def explore_actions(self):
        legal_actions: range = self.actions.get_legal_actions()
        random_action = np.random.choice(legal_actions)
        return random_action
    
    def act(self, state: np.ndarray, metrics: GameMetrics,
            verbose: bool = False):
        if np.random.random() < self.epsilon:  # Explore
            if verbose: print("[ACT] Explored")
            metrics.inc_num_exploration_steps()
            return self.explore_actions()
        else:  # Exploit
            if verbose: print("[ACT] Exploit")
            metrics.inc_num_exploitation_steps()
            return self.exploit_actions(state)
    
    def get_reward(self, game_status: int) -> int:
        if game_status == GOAL:
            reward = 1000
        elif game_status in [CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME]:
            reward = -1000
        else:
            reward = -1
        return reward
    
    def play_episode(self, game_metrics: GameMetrics, verbose: bool = False):
        # auxiliar structures:
        episode_buffer = list()
        # metrics:
        touched_ball = False
        passed_ball = False
        scored_goal = False
        # auxiliar:
        last_act = None
        while self.game_interface.in_game():
            if self.features.has_ball():
                touched_ball = True
            # Update environment features:
            features_array = self.features.get_features()
            # Act:
            act = self.act(features_array, metrics=game_metrics, verbose=False)
            if verbose:
                if act == last_act:
                    log_action = False
                else:
                    print(f"{self.features.team_ball_possession}; "
                          f"{self.features.has_ball()}")
                    log_action = True
                self.actions.execute_action(act, verbose=log_action)
                last_act = act
            else:
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
    
    def play(self, num_episodes: int, verbose: bool=False):
        """
        @param num_episodes: number of episodes to train in this iteration
        @raise ServerDownError
        @return: Game Metrics
        """
        experience_buffer = LearnBuffer()
        game_metrics = GameMetrics()
        
        for ep in range(num_episodes):
            # Check if server still running:
            try:
                self.game_interface.check_server_is_up()
            except ServerDownError:
                print("!!SERVER DOWN!! Test {}/{}".format(ep, num_episodes))
                return experience_buffer, game_metrics.export_to_dict()
            
            # Update features:
            self.features.re_calculate_features(
                observation=self.game_interface.get_observation(),
                last_player_touch_ball_uniform_num=0
            )
            # Play episode:
            ep_buffer, ep_metrics = self.play_episode(game_metrics, verbose)
            # Save episode:
            experience_buffer.save_episode(ep_buffer)
            
            # Update auxiliar variables:
            game_metrics.add_episode_metrics(
                ep_metrics, goal=self.game_interface.scored_goal())
            
            # Game Reset
            self.game_interface.reset()
        
        return experience_buffer, game_metrics.export_to_dict()
