#!/usr/bin/hfo_env python3
# encoding utf-8

from hfo import HFOEnvironment, HIGH_LEVEL_FEATURE_SET, QUIT, IN_GAME

from actions_levels import BaseActions
import settings


class HFOGoalkeeperPlayer(object):
    def __init__(self,
                 config_dir=settings.CONFIG_DIR, agent_id=0, port=6000,
                 server_addr='localhost', num_opponents=0, num_teammates=0):
        self.hfo = HFOEnvironment()
        self.config_dir = config_dir
        self.port = port
        self.server_addr = server_addr
        self.status = IN_GAME
    
    def reset(self) -> list:
        self.status = IN_GAME
        return self.hfo.getState()
    
    def connect_to_server(self):
        """ Establish connection with HFO server """
        self.hfo.connectToServer(
            HIGH_LEVEL_FEATURE_SET,
            self.config_dir,
            self.port,
            self.server_addr,
            team_name='base_right',
            play_goalie=True)
    
    def step(self, hfo_action: int, *args) -> (int, list):
        """
        Method that serves as an interface between a script controlling the
        agent and the environement_features. Method returns the current status
        of the episode and nextState
        """
        self.hfo.act(hfo_action, *args)
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