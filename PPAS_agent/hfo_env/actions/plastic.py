import random

import numpy as np
from hfo import MOVE_TO, MOVE, KICK_TO, NOOP, DRIBBLE, PASS, GO_TO_BALL, \
    INTERCEPT

from multi_agents.hfo_env.actions.base import Actions
from multi_agents.hfo_env.game_interface import GameInterface
from multi_agents.hfo_env.features.plastic import PlasticFeatures
from multi_agents.utils import get_opposite_vector


class PlasticActions(Actions):
    name = "plasticActions"
    
    def __init__(self, num_team: int, features: PlasticFeatures,
                 game_interface: GameInterface):
        super().__init__(num_team, features, game_interface)

    def _dribble_action(self, num_rep: int = 1):
        uni_number = self.game_interface.hfo.getUnum()
        for step in range(num_rep):
            if not self.game_interface.in_game():
                return
            # Has ball?
            elif self.features.has_ball():
                # Near opponent?
                if self.features.near_opponent(dist=0.15):
                    # Played a bit:
                    if step > 0:
                        return
                    # Haven't done any action:
                    else:
                        self._step(DRIBBLE)
                # Far from op:
                else:
                    self._step(DRIBBLE)
            # Kicked Ball?
            elif self.features.near_coords(self.features.ball_coord,
                                           self.features.agent_coord, ref=0.25):
                # The agent was the last to touch the ball:
                if self.game_interface.last_player_to_touch_ball == uni_number:
                    self._step(DRIBBLE)
                else:
                    return self._step(MOVE)
            # Far from ball:
            else:
                return self._step(MOVE)
        # End dribble reps:
        if self.features.has_ball():
            return
        else:
            while not self.features.has_ball():
                if not self.game_interface.in_game():
                    return
                # Near Ball:
                elif self.features.near_coords(self.features.ball_coord,
                                               self.features.agent_coord,
                                               ref=0.25):
                    # The agent was the last to touch the ball:
                    if self.game_interface.last_player_to_touch_ball == uni_number:
                        self._step(DRIBBLE)
                    else:
                        return self._step(MOVE)
                # Far from ball:
                else:
                    return self._step(MOVE)
            return
            

    def _move_to_goal(self, num_rep: int = 1):
        """ Move towards the opposing goal
        TODO define different points near goal and calculate the best position
        """
        # Goal area (x_range, y_range)
        goal_area = [[0.2, 1], [-0.7, 0.7]]
        for _ in range(num_rep):
            action = (MOVE_TO, 0.4, 0)
            backup_action = MOVE
            # Game over?
            if not self.game_interface.in_game():
                return
            # Has ball?
            elif self.features.has_ball():
                return self._step(DRIBBLE)
            # Teammate passed the ball, shoot or lost the ball?
            elif self.features.teammates_lost_ball:
                return self._step(backup_action)
            # Inside Goal area:
            elif 0.2 <= self.features.agent_coord[0] <= 1 and \
                    -0.6 <= self.features.agent_coord[0] <= 0.6:
                self._step(backup_action)
            # Goal to goal area:
            else:
                self._step(action)
        return

    def _move_to_ball(self, num_rep: int = 1):
        """ Move towards the ball"""
        if self.features.has_ball():
            raise ValueError("[action: move_to_ball] Agent already has ball!")
        
        for step in range(num_rep):
            backup_action = MOVE
            # Game over?
            if not self.game_interface.in_game():
                return
            # Agent has ball?
            elif self.features.has_ball():
                return self._step(DRIBBLE)
            # No one has ball:
            elif not self.features.teammates_have_ball():
                self._step(INTERCEPT)
            # Team has ball:
            else:
                # Near teammate?
                if self.features.near_coords(self.features.ball_coord,
                                             self.features.agent_coord,
                                             ref=0.2):
                    return self._step(MOVE)
                else:
                    self._step(INTERCEPT)
        return

    def _best_shoot_ball(self):
        """ Tries to shoot, if it fail, kicks to goal randomly """
        self._shoot()
        # If fail to shoot:
        if self.game_interface.in_game() and self.features.has_ball():
            self._kick_to_best_angle()
        return

    def _move_to_nearest_teammate(self, num_rep: int = 1):
        t_idx, _ = self.features.get_nearest_teammate_coord()
        for step in range(num_rep):
            t_coord = self.features.ts_coord[t_idx]
            action = (MOVE_TO, t_coord[0], t_coord[1])
            backup_action = MOVE
            # Game over?
            if not self.game_interface.in_game():
                return
            # Has ball?
            elif self.features.has_ball():
                return self._step(DRIBBLE)
            # Teammate passed the ball, shoot or lost the ball?
            elif self.features.teammates_lost_ball:
                return self._step(backup_action)
            # Still far from teammate:
            elif not self.features.near_coords(self.features.agent_coord,
                                               t_coord):
                self._step(action)
            # Near teammate, but near opponent? Move away a bit:
            elif self.features.near_opponent():
                self._step(backup_action)
            # Near teammate, but far from opponent:
            else:
                return self._do_nothing(1)
        return

    def _move_away_from_nearest_teammate(self, num_rep: int = 1):
        def get_x_y(vector):
            # Coordinates:
            x_pos = self.features.agent_coord[0] + vector[0]
            y_pos = self.features.agent_coord[1] + vector[1]
            if abs(x_pos) > 0.8:
                x_pos = 0.8 if x_pos > 0 else -0.8
            if abs(y_pos) > 0.8:
                y_pos = 0.8 if y_pos > 0 else -0.8
            return x_pos, y_pos
        
        t_idx, t_coord = self.features.get_nearest_teammate_coord()
        # opp_vector = get_opposite_vector(self.features.agent_coord, t_coord)

        for step in range(num_rep):
            # x, y = get_x_y(opp_vector)
            # action = (MOVE_TO, x, y)
            backup_action = MOVE
            # Game over?
            if not self.game_interface.in_game():
                return
            # Has ball?
            elif self.features.has_ball():
                return self._step(DRIBBLE)
            # Teammate passed the ball, shoot or lost the ball?
            elif self.features.teammates_lost_ball:
                return self._step(backup_action)
            elif abs(np.linalg.norm(self.features.agent_coord-t_coord)) > 0.4:
                return self._step(backup_action)
            else:
                self._step(backup_action)
        return

    def _move_to_nearest_opponent(self, num_rep: int = 1):
        op_idx, _ = self.features.get_nearest_opponent_coord()
        for step in range(num_rep):
            op_coord = self.features.opps_coord[op_idx]
            # action = (MOVE_TO, op_coord[0], op_coord[1])
            backup_action = MOVE
            # Game over?
            if not self.game_interface.in_game():
                return
            # Teammate passed the ball, shoot or lost the ball?
            elif self.features.teammates_lost_ball:
                return self._step(backup_action)
            # Has ball?
            elif self.features.has_ball():
                return self._step(DRIBBLE)
            # Still far from opponent:
            elif not self.features.near_coords(self.features.agent_coord,
                                               op_coord):
                self._step(backup_action)
            # Near opponent:
            else:
                return self._step(backup_action)
        else:
            return

    def _move_away_from_nearest_opponent(self, num_rep: int = 1):
        def get_x_y(vector):
            # Coordinates:
            x_pos = self.features.agent_coord[0] + vector[0]
            y_pos = self.features.agent_coord[1] + vector[1]
            if abs(x_pos) > 0.8:
                x_pos = 0.8 if x_pos > 0 else -0.8
            if abs(y_pos) > 0.8:
                y_pos = 0.8 if y_pos > 0 else -0.8
            return x_pos, y_pos
    
        for step in range(num_rep):
            # op_idx, near_op_coord = self.features.get_nearest_opponent_coord()
            # opp_vector = get_opposite_vector(self.features.agent_coord,
            #                                  near_op_coord)
            # x, y = get_x_y(opp_vector)
            # action = (MOVE_TO, x, y)
            backup_action = MOVE
            # Game over?
            if not self.game_interface.in_game():
                return
            # Has ball?
            elif self.features.has_ball():
                return self._step(DRIBBLE)
            # Teammate passed the ball, shoot or lost the ball?
            elif self.features.teammates_lost_ball:
                return self._step(backup_action)
            # Near opponent:
            elif self.features.near_opponent(dist=0.3):
                self._step(backup_action)
            else:
                self._step(backup_action)
        return

    def _pass_ball(self, teammate_id: int, verbose: bool=False):
        """ Tries to use the PASS action, if it fails, Kicks in the direction
        of the teammate"""
        uniform = self.features.teammates_uniform_numbers[teammate_id]
        for step in range(2):
            # Game over?
            if not self.game_interface.in_game():
                return
            # Lost has ball:
            elif not self.features.has_ball():
                return self._step(MOVE)
            else:
                hfo_action = (PASS, int(uniform))
                # Step game:
                self._step(action=hfo_action)
        # Failed the action PASS 2 times:
        else:
            if verbose:
                print(f"[PASS:{teammate_id}] Failed PASS to {uniform}; Will Kick")
            t_coord = self.features.ts_coord[teammate_id]
            hfo_action = (KICK_TO, t_coord[0], t_coord[1], 1.7)
            # Step game:
            self._step(action=hfo_action)
        return

    def execute_action(self, action_idx: int, verbose: bool = False):
        """ Receiving the idx of the action, the agent executes it and
        returns the game status """
        # Check action_idx:
        if not self.check_legal_action(action_idx):
            raise ValueError(f"[Actions] ILLEGAL ACTION {action_idx}. "
                             f"Agent has ball? {self.features.has_ball()}. "
                             f"Legal actions: {self.get_legal_actions()}")

        action_name = self.get_action_name(action_idx)
        if verbose:
            print(f"<ACTION> :: {action_name}")
        # Actions with ball:
        if action_name == "SHOOT":
            self._best_shoot_ball()
        elif action_name == "SHORT_DRIBBLE":
            self._dribble_action(self.N_SHORT_DRIBBLE_STEPS)
        elif action_name == "LONG_DRIBBLE":
            self._dribble_action(self.N_LONG_DRIBBLE_STEPS)
        elif "PASS" in action_name:
            _, teammate_id = action_name.split("PASS")
            self._pass_ball(int(teammate_id))
        # Actions without ball:
        elif action_name == "NOOP":
            self._do_nothing(self.N_NOOP_STEPS)
        elif action_name == "MOVE_TO_BALL":
            self._move_to_ball(self.N_GO_TO_BALL_STEPS)
        elif action_name == "MOVE_TO_GOAL":
            self._move_to_goal(self.N_MOVE_STEPS)
        elif action_name == "MOVE_TO_NEAR_TEAM":
            self._move_to_nearest_teammate(self.N_MOVE_STEPS)
        elif action_name == "MOVE_FROM_NEAR_TEAM":
            self._move_away_from_nearest_teammate(self.N_MOVE_STEPS)
        elif action_name == "MOVE_TO_NEAR_OP":
            self._move_to_nearest_opponent(self.N_MOVE_STEPS)
        elif action_name == "MOVE_FROM_NEAR_OP":
            self._move_away_from_nearest_opponent(self.N_MOVE_STEPS)
        else:
            raise ValueError(f"[Actions] Wrong name: {action_name}")
        # Check game integrety:
        self.game_interface.check_game_consistency()
