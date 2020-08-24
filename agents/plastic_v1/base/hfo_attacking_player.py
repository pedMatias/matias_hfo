#!/usr/bin/env python3
#encoding utf-8
from typing import Optional

from hfo import HFOEnvironment, HIGH_LEVEL_FEATURE_SET, QUIT, IN_GAME, \
    SERVER_DOWN, GOAL

from actions_levels import BaseActions
from agents.utils import ServerDownError
import settings


class HFOAttackingPlayer(object):
    def __init__(self, config_dir=settings.CONFIG_DIR, agent_id=0, port=6000,
                 server_addr='localhost', num_opponents=0, num_teammates=0):
        self.hfo = HFOEnvironment()
        self.config_dir = config_dir
        self.port = port
        self.server_addr = server_addr
        self.num_teammates = num_teammates
        self.num_opponents = num_opponents
        self.agent_id = agent_id
        self.episode = 0
        self.num_steps = 0
        self.status = IN_GAME
        
    def connect_to_server(self):
        """ Establish connection with HFO server """
        self.hfo.connectToServer(
            feature_set=HIGH_LEVEL_FEATURE_SET,
            config_dir=self.config_dir,
            server_port=self.port,
            server_addr=self.server_addr,
            team_name='base_left',
            play_goalie=False)
    
    def reset(self) -> list:
        self.status = IN_GAME
        self.episode += 1
        self.num_steps = 0
        return self.hfo.getState()
    
    def get_observation(self):
        return self.hfo.getState()

    def step(self, hfo_action) -> (int, list):
        """
        Method that serves as an interface between a script controlling the
        agent and the environment_features. Method returns the current status
        of the episode and nextState
        @param hfo_action: [int, tuple]
        """
        self.hfo.act(*hfo_action)
        self.num_steps += 1
        self.status = self.hfo.step()
        return self.status, self.hfo.getState()

    def quit_game(self):
        self.hfo.act(QUIT)
    
    def get_game_status(self) -> int:
        return self.status
    
    def in_game(self) -> bool:
        if self.status == IN_GAME:
            return True
        else:
            return False
    
    def scored_goal(self) -> bool:
        return self.status == GOAL
    
    def check_server_is_up(self):
        if self.hfo.step() == SERVER_DOWN:
            raise ServerDownError("Server is Down!!")
