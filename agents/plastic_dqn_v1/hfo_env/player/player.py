# !/usr/bin/hfo_env python3
# encoding utf-8
from collections import namedtuple

from hfo import GOAL, CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS
import numpy as np

from agents.utils import ServerDownError
from agents.plastic_dqn_v1.hfo_env.game_interface import HFOAttackingPlayer
from agents.plastic_dqn_v1.agent.dqn import DQN
from agents.plastic_dqn_v1.agent.replay_buffer import LearnBuffer, Transition
from agents.plastic_dqn_v1.hfo_env.actions.complex import Actions
from agents.plastic_dqn_v1.hfo_env.features.plastic_features import PlasticFeatures

"""
This module is used to test the agent previous trained
"""

EpisodeMetrics = namedtuple(
    'EpisodeMetrics',
    ['touched_ball', 'passed_ball', 'num_correct_actions', 'num_wrong_actions']
)


class GameMetrics:
    def __init__(self):
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
    
    def inc_num_wrong_actions(self):
        self.num_wrong_actions += 1
    
    def inc_num_correct_actions(self):
        self.num_correct_actions += 1

    def inc_num_games_touched_ball(self):
        self.num_games_touched_ball += 1

    def inc_num_games_passed_ball(self):
        self.num_games_passed_ball += 1
    
    def add_episode_metrics(self, ep_metrics: EpisodeMetrics, goal: bool):
        self.inc_num_ep()
        if goal:
            self.inc_num_wins()
        if ep_metrics.touched_ball:
            self.inc_num_games_touched_ball()
        if ep_metrics.passed_ball:
            self.inc_num_games_passed_ball()
        
        self.num_correct_actions += ep_metrics.num_correct_actions
        self.num_wrong_actions += ep_metrics.num_wrong_actions
    
    def export(self, verbose: bool = True):
        win_rate = self.num_wins / self.num_ep
        # Correct actions rate:
        total_num_actions = self.num_correct_actions + self.num_wrong_actions
        correct_actions_rate = self.num_correct_actions / total_num_actions
        correct_actions_rate = round(correct_actions_rate, 2)
        # Touch ball rate:
        avr_touched_ball_rate = self.num_games_touched_ball / self.num_ep
        avr_touched_ball_rate = round(avr_touched_ball_rate, 2)
        # Pass ball rate:
        aux = self.num_games_passed_ball / max(self.num_games_touched_ball, 1)
        avr_passed_ball_rate = round(aux, 2)
        if verbose:
            print(f"[Game Metrics: Summary] WIN rate = {win_rate}; "
                  f"Correct Actions Rate = {correct_actions_rate}; "
                  f"Touched Ball rate = {avr_touched_ball_rate}; "
                  f"Pass Ball rate={avr_passed_ball_rate};")
        return win_rate, correct_actions_rate, \
            avr_touched_ball_rate, avr_passed_ball_rate
    
    def export_to_dict(self, verbose: bool = True):
        win_rt, corr_act_rt, avr_ball_rt, avr_pass_rt = \
            self.export(verbose=verbose)
        return dict(win_rate=win_rt,
                    correct_actions_rate=corr_act_rt,
                    avr_touched_ball_rate=avr_ball_rt,
                    avr_passed_ball_rate=avr_pass_rt)


class Player:
    def __init__(self, num_opponents: int, num_teammates: int, model_file: str,
                 epsilon: int = 1, port: int = 6000):
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
    
    def exploit_actions(self, state: np.ndarray, verbose: bool = False) -> int:
        q_predict = self.dqn.predict(state)
        max_list = np.where(q_predict == q_predict.max())
        action = np.random.choice(max_list[0]) if len(max_list[0]) > 1 \
            else np.argmax(q_predict)
        if verbose:
            print("Q values {} -> {}".format(q_predict, int(action)))
        return int(action)
    
    def explore_actions(self):
        random_action = np.random.randint(0, self.actions.num_actions)
        return random_action
    
    def act(self, state: np.ndarray, verbose: bool = False):
        if np.random.random() < self.epsilon:  # Explore
            if verbose: print("[ACT] Explored")
            return self.explore_actions()
        else:  # Exploit
            if verbose: print("[ACT] Exploit")
            return self.exploit_actions(state)
    
    def get_reward(self, game_status: int, correct_action: bool) -> int:
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
    
    def play_episode(self):
        # auxiliar structures:
        episode_buffer = list()
        # metrics:
        touched_ball = False
        passed_ball = False
        num_wrong_actions = 0
        num_correct_actions = 0
        while self.game_interface.in_game():
            touched_ball = True if self.features.has_ball() else False
            # Update environment features:
            features_array = self.features.get_features()
            # Act:
            act = self.act(features_array, verbose=False)
            status, correct_action, passed_ball_succ = \
                self.actions.execute_action(act)

            # Store transition:
            # (obs, action, reward, new obs, done?)
            transition = Transition(
                obs=features_array,
                act=act,
                reward=self.get_reward(status, correct_action),
                new_obs=self.features.get_features(),
                done=not self.game_interface.in_game(),
                correct_action=correct_action
            )
            episode_buffer.append(transition)
            
            # Metrics:
            passed_ball = True if passed_ball_succ else False
            if correct_action:
                num_correct_actions += 1
            else:
                num_wrong_actions += 1
        metrics = EpisodeMetrics(
            touched_ball=touched_ball,
            passed_ball=passed_ball,
            num_wrong_actions=num_wrong_actions,
            num_correct_actions=num_correct_actions)
        return episode_buffer, metrics
    
    def play(self, num_episodes: int, starts_fixed_position: bool = True):
        """
        @param num_episodes: number of episodes to train in this iteration
        @param starts_fixed_position: bool
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
            self.features.update_features(
                observation=self.game_interface.get_observation())
            
            # Play episode:
            ep_buffer, ep_metrics = self.play_episode()
            # Save episode:
            experience_buffer.save_episode(ep_buffer, verbose=True)
            
            # Update auxiliar variables:
            game_metrics.add_episode_metrics(
                ep_metrics, goal=self.game_interface.scored_goal())
            
            # Game Reset
            self.game_interface.reset()
        
        return experience_buffer, game_metrics.export_to_dict()
