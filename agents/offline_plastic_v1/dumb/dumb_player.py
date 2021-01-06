#!/usr/bin/hfo_env python3
# encoding utf-8
import argparse
import random

import numpy as np
from hfo import GOAL, IN_GAME, CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS

import settings
from agents.utils import ServerDownError, get_vertices_around_ball
from agents.plastic_v1.base.hfo_attacking_player import HFOAttackingPlayer
from agents.plastic_v1.deep_agent import DQNAgent
from agents.plastic_v1.actions.v1 import Actions
from agents.plastic_v1.features.plastic_features import PlasticFeatures

STARTING_POSITIONS = {"TOP LEFT": (-0.5, -0.6), "TOP RIGHT": (0.3, -0.6),
                      "MID LEFT": (-0.5, 0.0), "MID RIGHT": (0.4, 0.0),
                      "BOTTOM LEFT": (-0.5, 0.6), "BOTTOM RIGHT": (0.3, 0.6)}


class Player:
    def __init__(self, num_opponents: int, num_teammates: int,
                 port: int = 6000):
        # Features Interface:
        self.features = PlasticFeatures(num_op=num_opponents,
                                        num_team=num_teammates)
        # Game Interface:
        self.game_interface = HFOAttackingPlayer(num_opponents=num_opponents,
                                                 num_teammates=num_teammates,
                                                 port=port)
        self.game_interface.connect_to_server()
        # Actions Interface:
        self.actions = Actions(num_team=num_teammates, features=self.features,
                               game_interface=self.game_interface)
        # Agent instance:
        self.agent = DQNAgent(num_features=self.features.num_features,
                              num_actions=self.actions.get_num_actions(),
                              learning_rate=0.005, discount_factor=0.99,
                              epsilon=1, final_epsilon=0.001,
                              epsilon_decay=0.99997, tau=0.125)
        # Auxiliar attributes
        self.starting_pos_list = list(STARTING_POSITIONS.values())
        self.num_ep = 0
    
    def get_reward(self, game_status: int, correct_action: bool) -> int:
        reward = 0
        if correct_action:
            reward += 1
        else:
            reward -= 1
        
        if game_status == GOAL:
            reward += 1000
        elif game_status in [CAPTURED_BY_DEFENSE, OUT_OF_BOUNDS, OUT_OF_TIME]:
            reward -= 1000
        else:
            reward -= 1
        return reward
    
    def set_starting_game_conditions(self, start_with_ball: bool = True,
                                     start_pos: tuple = None,
                                     starts_fixed_position: bool = True,
                                     verbose: bool = False):
        """
        Set starting game conditions. Move for initial position, for example
        """
        if start_with_ball:
            if starts_fixed_position:
                if not start_pos:
                    aux_idx = self.num_ep % len(self.starting_pos_list)
                    start_pos = self.starting_pos_list[aux_idx]
                self.actions.dribble_to_pos(start_pos, stop=True)
                if verbose:
                    print(f"[START GAME] Ball; {start_pos}")
            else:
                while not self.features.has_ball():
                    self.actions.move_to_ball()
                if verbose:
                    print(f"[START GAME] Ball; RANDOM")
        
        else:
            if starts_fixed_position:
                if not start_pos:
                    ball_pos: list = list(self.features.get_ball_coord())
                    starting_corners = get_vertices_around_ball(ball_pos)
                    start_pos = random.choice(starting_corners)
                self.actions.move_to_pos(start_pos)
                if verbose:
                    print(f"[START GAME] NO Ball; {start_pos}")
            else:
                # Start in current position
                if verbose:
                    print(f"[START GAME] NO Ball; RANDOM")
                pass
        # Informs the other players that it is ready to start:
        self.game_interface.hfo.say(settings.PLAYER_READY_MSG)
    
    def test(self, num_episodes: int, start_with_ball: bool = True,
             starts_fixed_position: bool = True) -> float:
        """
        @param num_episodes: number of episodes to run
        @param start_with_ball: flag
        @param starts_fixed_position: flag
        @return: (float) the win rate
        """
        # metrics variables:
        _num_wins = 0
        self.num_ep = 0
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
            
            # Go to origin position:
            print(f"\n[TEST] {ep}/{num_episodes}")
            self.set_starting_game_conditions(
                start_with_ball=start_with_ball, start_pos=None,
                starts_fixed_position=starts_fixed_position, verbose=True)
            
            # Start learning loop
            prev_act = -1
            while self.game_interface.in_game():
                # Update environment features:
                features_array = self.features.get_features().copy()
                # Act:
                act = self.agent.exploit_actions(features_array)
                if act != prev_act:
                    self.actions.execute_action(act, verbose=True)
                else:
                    self.actions.execute_action(act)
                prev_act = act
            
            # Update auxiliar variables:
            if self.game_interface.scored_goal():
                print("[GOAL]")
                _num_wins += 1
            else:
                print("[FAIL]")
            # Game Reset
            self.game_interface.reset()
            self.num_ep += 1
        avr_win_rate = round(_num_wins / num_episodes, 2)
        print("[TEST: Summary] WIN rate = {};".format(avr_win_rate))
        return avr_win_rate
    
    def train(self, num_train_episodes: int, num_total_train_ep: int,
              starts_fixed_position: bool = True, start_with_ball: bool = True):
        """
        @param num_train_episodes: number of episodes to train in this iteration
        @param num_total_train_ep: number total of episodes to train
        @param starts_fixed_position: bool
        @param start_with_ball: bool
        @raise ServerDownError
        @return: (QLearningAgentV5) the agent
        """
        # metrics variables:
        _num_wins = 0
        _sum_epsilons = 0
        self.num_ep = 0
        
        game_buffer = []
        actions = [[9, 8, 7], [9, 9, 9, 8, 7],
                   [7], [9, 9, 7],
                   [9, 9, 7], [9, 9, 9, 8, 7],
                   [9, 9, 7], [9, 9, 9, 7],
                   [8, 7], [9, 8, 7],
                   [9, 8, 8, 7], [9, 9, 8, 8, 7]]
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
                start_with_ball=start_with_ball, start_pos=None,
                starts_fixed_position=starts_fixed_position)
    
            # Start learning loop
            status = IN_GAME
            episode_buffer = list()
            correct_action = False
    
            ep_actions = actions.pop()
            while self.game_interface.in_game():
                if ep_actions is None or len(ep_actions) == 0:
                    act = 0
                else:
                    act = ep_actions[0]
                ep_actions = ep_actions[1:]
        
                # Update environment features:
                features_array = self.features.get_features().copy()
        
                # Act:
                # act = self.agent.act(features_array)
                status, correct_action = self.actions.execute_action(
                    act, verbose=True)
        
                # Every step we update replay memory and train main network
                done = not self.game_interface.in_game()
                # Store transition:
                # (obs, action, reward, new obs, done?)
                transition = np.array(
                    [features_array,
                     act,
                     self.get_reward(status, correct_action),
                     self.features.get_features(),
                     done])
                episode_buffer.append(transition)
        
                # Train:
                # self.agent.train(terminal_state=done)
            
            # TODO remove:
            game_buffer += episode_buffer
            if self.game_interface.scored_goal() or status == GOAL:
                print("GOAL!!")
            #     _num_wins += 1
            #     reward = self.get_reward(GOAL, correct_action)
            # else:
            #     reward = self.get_reward(status, correct_action)
            # Add episodes:
            # self.agent.store_episode(episode_buffer)
            # Update auxiliar variables:
            _sum_epsilons += self.agent.epsilon
            # Update Agent:
            self.agent.restart(num_total_train_ep)
            # Game Reset
            self.game_interface.reset()
            self.num_ep += 1
        print(game_buffer)
        avr_epsilon = round(_sum_epsilons / num_train_episodes, 3)
        print("[TRAIN: Summary] WIN rate = {}; AVR epsilon = {}".format(
            _num_wins / num_train_episodes, avr_epsilon))
        return avr_epsilon


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6000)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=0)
    parser.add_argument('--starts_with_ball', type=bool, default=True)
    
    # Parse Arguments:
    args = parser.parse_args()
    port = args.port
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_episodes
    starts_with_ball = args.starts_with_ball
    
    # Start Player:
    player = Player(num_teammates=num_team, num_opponents=num_op, port=port)
    
    print("\n[PLAYER] num_opponents={}; num_teammates={}; "
          "start_with_ball={}".format(num_op, num_op, starts_with_ball))
    
    player.train(num_train_episodes=num_episodes,
                 num_total_train_ep=num_episodes,
                 start_with_ball=starts_with_ball)
    
    print("\n\n!!!!!!!!! AGENT EXIT !!!!!!!!!!!!\n\n")