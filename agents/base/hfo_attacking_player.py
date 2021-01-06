#!/usr/bin/hfo_env python3
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
    
    def reset(self) -> list:
        self.status = IN_GAME
        self.episode += 1
        self.num_steps = 0
        return self.hfo.getState()
    
    def get_state(self):
        # TODO deprecated
        return self.hfo.getState()
    
    def get_observation_array(self):
        return self.hfo.getState()

    def connect_to_server(self):
        """ Establish connection with HFO server """
        self.hfo.connectToServer(
            HIGH_LEVEL_FEATURE_SET,
            self.config_dir,
            self.port,
            self.server_addr,
            team_name='base_left',
            play_goalie=False)

    def step(self, hfo_action, has_ball: bool, filter=True) -> (int, list):
        """
        Method that serves as an interface between a script controlling the
        agent and the environment_features. Method returns the current status
        of the episode and nextState
        @param hfo_action: [int, tuple]
        @param has_ball:
        @param filter:
        """
        if filter:
            if type(hfo_action) is tuple:
                action = hfo_action[0]
                params = hfo_action[1:]
            else:
                action = hfo_action
                params = ()
            action_args = BaseActions.ActionManager.valid_action(
                action, has_ball, params)
        else:
            action_args = hfo_action
        self.hfo.act(*action_args)
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
