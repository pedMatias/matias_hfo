# !/usr/bin/hfo_env python3
# encoding utf-8
import os
from collections import namedtuple


EpisodeMetrics = namedtuple(
    'EpisodeMetrics', ['touched_ball', 'passed_ball']
)


class GameMetrics:
    def __init__(self):
        self.num_ep = 0
        self.num_wins = 0
        self.num_games_touched_ball = 0
        self.num_games_passed_ball = 0
        # Exploration - Exploitation:
        self.guessed_teams = list()
        self.correct_team = 0
        self.teams = list
    
    def inc_num_ep(self):
        self.num_ep += 1
    
    def inc_num_wins(self):
        self.num_wins += 1
    
    def inc_num_games_touched_ball(self):
        self.num_games_touched_ball += 1
    
    def inc_num_games_passed_ball(self):
        self.num_games_passed_ball += 1
    
    def append_guessed_teams(self, guessed_teams: list):
        self.guessed_teams.append(guessed_teams)
    
    def set_correct_team(self, team_name: str):
        self.correct_team = team_name
    
    def add_episode_metrics(self, ep_metrics: EpisodeMetrics, goal: bool,
                            guessed_teams: list):
        self.inc_num_ep()
        if goal:
            self.inc_num_wins()
        if ep_metrics.touched_ball:
            self.inc_num_games_touched_ball()
        if ep_metrics.passed_ball:
            self.inc_num_games_passed_ball()
        
        # Guessed teams:
        self.append_guessed_teams(guessed_teams)
    
    def export(self, verbose: bool = True):
        win_rate = self.num_wins / self.num_ep
        
        # Touch ball rate:
        avr_touched_ball_rate = self.num_games_touched_ball / self.num_ep
        avr_touched_ball_rate = round(avr_touched_ball_rate, 2)
        
        # Pass ball rate:
        aux = self.num_games_passed_ball / max(self.num_games_touched_ball, 1)
        avr_passed_ball_rate = round(aux, 2)
        
        # Guessed correct team:
        total_correct_perct = 0
        for ep_guesses in self.guessed_teams:
            total_correct_perct += ep_guesses.count(self.correct_team) / \
                                   len(ep_guesses)
        total_correct_perct = total_correct_perct / len(self.guessed_teams)
        
        if verbose:
            print(f"[Game Metrics: Summary] WIN rate = {win_rate}; "
                  f"Touched Ball rate = {avr_touched_ball_rate}; "
                  f"Pass Ball rate={avr_passed_ball_rate}; "
                  f"Correct_guess_perct={total_correct_perct};")
        return win_rate, avr_touched_ball_rate, avr_passed_ball_rate, \
               total_correct_perct
    
    def export_to_dict(self, verbose: bool = True):
        win_rt, avr_ball_rt, avr_pass_rt, corr_guess = \
            self.export(verbose=verbose)
        return dict(win_rate=win_rt,
                    avr_touched_ball_rate=avr_ball_rt,
                    avr_passed_ball_rate=avr_pass_rt,
                    total_correct_perct=corr_guess)