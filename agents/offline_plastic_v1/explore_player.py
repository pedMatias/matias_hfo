#!/usr/bin/env python3
# encoding utf-8
import argparse
import json
import os
import pickle
import random
from copy import copy
from typing import List

import numpy as np
from hfo import GOAL, IN_GAME, CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS

import settings
from agents.utils import ServerDownError, get_vertices_around_ball
from agents.offline_plastic_v1.base.hfo_attacking_player import \
    HFOAttackingPlayer
from agents.offline_plastic_v1.deep_agent import DQNAgent, Transition
from agents.offline_plastic_v1.actions.complex import Actions
from agents.offline_plastic_v1.features.plastic_features import \
    PlasticFeatures, HAS_BALL_FEATURE_WEIGHT
from agents.offline_plastic_v1.aux import print_transiction, mkdir

STARTING_POSITIONS = {"TOP LEFT": (-0.5, -0.7), "TOP RIGHT": (0.4, -0.7),
                      "MID LEFT": (-0.5, 0.0), "MID RIGHT": (0.4, 0.0),
                      "BOTTOM LEFT": (-0.5, 0.7), "BOTTOM RIGHT": (0.4, 0.7)}


class Player:
    def __init__(self, num_opponents: int, num_teammates: int,
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
        self.agent = DQNAgent(num_features=self.features.num_features,
                              num_actions=self.actions.get_num_actions(),
                              epsilon=epsilon, create_model=False)
        # Auxiliar attributes
        self.starting_pos_list = list(STARTING_POSITIONS.values())
        self.num_ep = 0
    
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
                reward -= 1
        return reward

    def parse_episode(self, episodes_transitions: List[Transition],
                      verbose: bool = False) -> list:
        if len(episodes_transitions) == 0:
            return []
        
        # Remove last actions without ball:
        last_reward = copy(episodes_transitions[-1].reward)
        for idx in range(len(episodes_transitions) - 1, -1, -1):
            # Has ball:
            if episodes_transitions[idx].obs[5] > 0:
                episodes_transitions = episodes_transitions[:idx + 1]
                break
            # No ball:
            elif episodes_transitions[idx].obs[5] < 0:
                pass
            else:
                raise ValueError("Features has ball, wrong value!!")
        else:
            return []
            
        # selected wrong action?:
        if episodes_transitions[-1].correct_action is False and last_reward > 0:
            episodes_transitions[-1].reward = -1
        else:
            episodes_transitions[-1].reward = last_reward
        episodes_transitions[-1].done = True
        
        if verbose and random.random() > 0.99:
            print("\n ** Transictions:")
            for el in episodes_transitions:
                print_transiction(el.to_tuple(), self.actions, simplex=True)
            print('**')
        
        return episodes_transitions
    
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

    def explore(self, num_episodes: int,
                starts_fixed_position: bool = True,
                start_with_ball: bool = True):
        """
        @param num_episodes: number of episodes to train in this iteration
        @param starts_fixed_position: bool
        @param start_with_ball: bool
        @raise ServerDownError
        @return: (QLearningAgentV5) the agent
        """
        learn_episodes = []
        # metrics variables:
        _num_wins = 0
        _num_games_touched_ball = 0
        _num_games_passed_ball = 0
        
        self.num_ep = 0
        for ep in range(num_episodes):
            # Check if server still running:
            try:
                self.game_interface.check_server_is_up()
            except ServerDownError:
                print("!!SERVER DOWN!! TRAIN {}/{}".format(ep, num_episodes))
                exploration_rate = self.agent.num_explorations / (
                        self.agent.num_explorations +
                        self.agent.num_exploitations)
                avr_touched_ball_rate = \
                    round(_num_games_touched_ball / num_episodes, 2)
                avr_passed_ball_rate = round(
                    _num_games_passed_ball / max(_num_games_touched_ball, 1),
                    2)
                return learn_episodes, exploration_rate, \
                    avr_touched_ball_rate, avr_passed_ball_rate
            # Update features:
            self.features.update_features(self.game_interface.get_observation())
            
            # Go to origin position:
            self.set_starting_game_conditions(
                start_with_ball=start_with_ball, start_pos=None,
                starts_fixed_position=starts_fixed_position)
            
            # Start learning loop
            status = IN_GAME
            episode_buffer = list()
            # metrics:
            touched_ball = False
            passed_ball = False
            while self.game_interface.in_game():
                # Metrics: Touched ball:
                if self.features.has_ball():
                    touched_ball = True
                    
                # Update environment features:
                features_array = self.features.get_features()
                # Act:
                act = self.agent.act(features_array, verbose=False)
                
                status, correct_action, passed_ball_succ = \
                    self.actions.execute_action(act)

                # Save metric pass:
                if passed_ball_succ is True:
                    passed_ball = True
                    
                # Every step we update replay memory and train main network
                done = not self.game_interface.in_game()
                # Store transition:
                # (obs, action, reward, new obs, done?)
                transition = Transition(
                    obs=features_array,
                    act=act,
                    reward=self.get_reward(status, correct_action),
                    new_obs=self.features.get_features(),
                    done=done,
                    correct_action=correct_action
                )
                episode_buffer.append(transition)
                    
            if self.game_interface.scored_goal() or status == GOAL:
                try:
                    if episode_buffer[-1].reward != 1000:
                        raise Exception("Last action reward is wrong!")
                    _num_wins += 1
                except IndexError as e:
                    print("Episode Buffer Empty: ", episode_buffer)

            # Update auxiliar variables:
            if touched_ball:
                _num_games_touched_ball += 1
            if passed_ball:
                _num_games_passed_ball += 1
                    
            # Add episodes:
            episode_buffer = self.parse_episode(episode_buffer, verbose=True)
            learn_episodes += episode_buffer
            # Game Reset
            self.game_interface.reset()
            self.num_ep += 1
            
        exploration_rate = self.agent.num_explorations / \
            (self.agent.num_explorations + self.agent.num_exploitations)
        avr_touched_ball_rate = \
            round(_num_games_touched_ball / num_episodes, 2)
        avr_passed_ball_rate = \
            round(_num_games_passed_ball / max(_num_games_touched_ball, 1), 2)
        print("[EXPLORATION: Summary] WIN rate = {}; Exploration Rate = {}; "
              "Touched Ball rate = {}; Pass Ball rate={};".
              format(_num_wins / num_episodes, exploration_rate,
                     avr_touched_ball_rate, avr_passed_ball_rate))
        return learn_episodes, exploration_rate, avr_touched_ball_rate, \
            avr_passed_ball_rate
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--teammate_type', type=str, default=None)
    parser.add_argument('--opponent_type', type=str, default=None)
    parser.add_argument('--starts_with_ball', type=str, default="true")
    parser.add_argument('--starts_fixed_position', type=str, default="true")
    parser.add_argument('--epsilon', type=float, default=1)
    parser.add_argument('--stage', type=float)
    parser.add_argument('--dir', type=str)
    parser.add_argument('--port', type=int, default=6000)
    
    # Parse Arguments:
    args = parser.parse_args()
    print(f"\n[Player: set-up] {args}\n")
    
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_episodes = args.num_episodes
    op_type = args.opponent_type
    team_type = args.teammate_type
    directory = args.dir
    epsilon = args.epsilon
    stage = args.stage
    starts_with_ball = True if args.starts_with_ball == "true" else False
    starts_fixed_position = True if args.starts_fixed_position == "true" \
        else False
    port = args.port
    
    # Start Player:
    player = Player(num_teammates=num_team, num_opponents=num_op, port=port,
                    epsilon=epsilon)

    # Load Model if stage higher than 1:
    if stage > 1.0:
        # Beginning of a stage:
        if stage % 1 == 0:
            prev_sub_stage = int(stage - 1)
        else:
            prev_sub_stage = stage - 0.1
            prev_sub_stage = round(prev_sub_stage, 1)

        # Get model file:
        model_file = os.path.join(directory, f"agent_model_{prev_sub_stage}")

        if not os.path.isfile(model_file):
            print("[LOAD MODEL File] Cant find previous Model!!")
            pass
        else:
            print("[LOAD MODEL File] Load model {}".format(model_file))
            player.agent.load_model(model_file)
    
    # Explore game:
    learn_buffer, avr_exploration, avr_touched_ball_rate, \
        avr_passed_ball_rate = player.explore(
            num_episodes=num_episodes,
            start_with_ball=starts_with_ball,
            starts_fixed_position=starts_fixed_position)
    
    # Export train_data:
    train_data_file = os.path.join(directory, f"learn_buffer_{stage}")
    with open(train_data_file, "wb") as fp:
        pickle.dump(learn_buffer, fp)
    
    # Export metrics:
    data = {"number_episodes": num_episodes, "num_op": num_op,
            "op_type": op_type, "num_team": num_team, "team_type": team_type,
            "starts_with_ball": starts_with_ball,
            "starts_fixed_position": starts_fixed_position,
            "exploration_rate": avr_exploration,
            "avr_touched_ball_rate": avr_touched_ball_rate,
            "avr_passed_ball_rate": avr_passed_ball_rate}
    metrics_file = os.path.join(directory, f"exploration_metrics_{stage}.json")
    with open(metrics_file, 'w+') as fp:
        json.dump(data, fp)
        
    print("\n\n!!!!!!!!! Exploration Ended  !!!!!!!!!!!!\n\n")
