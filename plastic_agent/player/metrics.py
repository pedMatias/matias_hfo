# !/usr/bin/hfo_env python3
# encoding utf-8
from collections import namedtuple


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
        # Exploration - Exploitation:
        self.num_exploration_steps = 0
        self.num_exploitation_steps = 0
    
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
    
    def inc_num_exploration_steps(self):
        self.num_exploration_steps += 1
    
    def inc_num_exploitation_steps(self):
        self.num_exploitation_steps += 1
    
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
        
        # Exploration rate:
        if self.num_exploration_steps > 0:
            exploration_rt = self.num_exploration_steps / \
                (self.num_exploration_steps + self.num_exploitation_steps)
        else:
            exploration_rt = 0
        if verbose:
            print(f"[Game Metrics: Summary] WIN rate = {win_rate}; "
                  f"Correct Actions Rate = {correct_actions_rate}; "
                  f"Touched Ball rate = {avr_touched_ball_rate}; "
                  f"Pass Ball rate={avr_passed_ball_rate}; "
                  f"Exploration rate={exploration_rt};")
        return win_rate, correct_actions_rate, \
            avr_touched_ball_rate, avr_passed_ball_rate, exploration_rt
    
    def export_to_dict(self, verbose: bool = True):
        win_rt, corr_act_rt, avr_ball_rt, avr_pass_rt, exploration_rt = \
            self.export(verbose=verbose)
        return dict(win_rate=win_rt,
                    correct_actions_rate=corr_act_rt,
                    avr_touched_ball_rate=avr_ball_rt,
                    avr_passed_ball_rate=avr_pass_rt,
                    exploration_rate=exploration_rt)
