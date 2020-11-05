#!/usr/bin/hfo_env python3
#encoding utf-8

import hfo
import settings
from agents.utils import ServerDownError


class GameInterface(object):
    def __init__(self, agent_id=0, port=6000, server_addr='localhost',
                 num_opponents=0, num_teammates=0):
        # Hfo game interface:
        self.hfo = hfo.HFOEnvironment()
        # Server configuration:
        self.feature_set = hfo.HIGH_LEVEL_FEATURE_SET
        self.config_dir = settings.CONFIG_DIR
        self.port = port
        self.server_addr = server_addr
        self.team_name = 'base_left'
        self.play_goalie = False
        # Attributes:
        self.num_teammates = num_teammates
        self.num_opponents = num_opponents
        self.agent_id = agent_id
        # Metrics:
        self.episode = 0
        self.num_steps = 0
        self.status = hfo.IN_GAME
        
    def connect_to_server(self):
        """ Establish connection with HFO server """
        self.hfo.connectToServer(
            feature_set=self.feature_set,
            config_dir=self.config_dir,
            server_port=self.port,
            server_addr=self.server_addr,
            team_name=self.team_name,
            play_goalie=self.play_goalie)
    
    def reset(self) -> list:
        self.status = hfo.IN_GAME
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
        if isinstance(hfo_action, tuple):
            self.hfo.act(*hfo_action)
        else:
            self.hfo.act(hfo_action)
        self.num_steps += 1
        self.status = self.hfo.step()
        return self.status, self.hfo.getState()

    def quit_game(self):
        self.hfo.act(hfo.QUIT)
    
    def get_game_status(self) -> int:
        return self.status
    
    def in_game(self) -> bool:
        if self.status == hfo.IN_GAME:
            return True
        else:
            return False
    
    def scored_goal(self) -> bool:
        return self.status == hfo.GOAL
    
    def check_server_is_up(self):
        if self.hfo.step() == hfo.SERVER_DOWN:
            raise ServerDownError("Server is Down!!")
