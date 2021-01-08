# !/usr/bin/hfo_env python3
# encoding utf-8
from hfo import GOAL, CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS
import numpy as np

from agents.utils import ServerDownError
from plastic_agent.agent.dqn import DQN
from plastic_agent.agent.replay_buffer import LearnBuffer, Transition
from plastic_agent.player.metrics import GameMetrics, EpisodeMetrics
from plastic_agent.hfo_env.game_interface import GameInterface
from plastic_agent.hfo_env.features.plastic import PlasticFeatures
from plastic_agent.hfo_env.actions.plastic import PlasticActions

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
    
    def play_episode(self, game_metrics: GameMetrics):
        # auxiliar structures:
        episode_buffer = list()
        # metrics:
        touched_ball = False
        passed_ball = False
        num_wrong_actions = 0
        num_correct_actions = 0
        while self.game_interface.in_game():
            print("BALL?? ", self.game_interface.hfo.playerOnBall().unum)
            if self.features.has_ball():
                touched_ball = True
            # Update environment features:
            features_array = self.features.get_features()
            # Act:
            act = self.act(features_array, metrics=game_metrics, verbose=False)
            status, correct_action, passed_ball_succ = \
                self.actions.execute_action(act, verbose=False)

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
        return episode_buffer, metrics
    
    def play(self, num_episodes: int):
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
            self.features.update_features(
                observation=self.game_interface.get_observation())
            # Play episode:
            ep_buffer, ep_metrics = self.play_episode(game_metrics)
            # Save episode:
            experience_buffer.save_episode(ep_buffer, verbose=True)
            
            # Update auxiliar variables:
            game_metrics.add_episode_metrics(
                ep_metrics, goal=self.game_interface.scored_goal())
            
            # Game Reset
            self.game_interface.reset()
        
        return experience_buffer, game_metrics.export_to_dict()
