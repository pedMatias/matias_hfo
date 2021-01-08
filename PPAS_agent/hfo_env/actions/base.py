import numpy as np
from hfo import MOVE_TO, MOVE, KICK_TO, NOOP, DRIBBLE, PASS, GO_TO_BALL, \
    INTERCEPT, SHOOT

from multi_agents.hfo_env.game_interface import GameInterface
from multi_agents.hfo_env.features.plastic import PlasticFeatures
from multi_agents.utils import get_angle


class Actions:
    # ACTIONS:
    ACTIONS_WITHOUT_BALL = ["NOOP", "MOVE_TO_BALL", "MOVE_TO_GOAL",
                            "MOVE_TO_NEAR_TEAM", "MOVE_FROM_NEAR_TEAM",
                            "MOVE_TO_NEAR_OP", "MOVE_FROM_NEAR_OP"]
    ACTIONS_WITH_BALL = ["SHOOT", "SHORT_DRIBBLE", "LONG_DRIBBLE"]
    
    # Movement steps:
    N_SHORT_DRIBBLE_STEPS = 4
    N_LONG_DRIBBLE_STEPS = 12
    N_GO_TO_BALL_STEPS = 8
    N_MOVE_STEPS = 4
    N_NOOP_STEPS = 2
    
    # Shoot angles:
    shoot_possible_coord = [np.array([0.83, -0.17]), np.array([0.83, 0]),
                            np.array([0.83, 0.17])]
    
    def __init__(self, num_team: int, features: PlasticFeatures,
                 game_interface: GameInterface):
        self.num_teammates = num_team
        self.features = features
        self.game_interface = game_interface
        
        self._actions = list()
        self._actions_with_ball_idxs: range
        self._actions_without_ball_idxs: range
        # Actions without ball:
        self._set_up_actions_without_ball()
        # Actions with ball:
        self._set_up_actions_with_ball(num_team=num_team)
        
        self.num_actions = len(self._actions)
        
        print(f"[ACTIONS] num_team={num_team}, num-action={self.num_actions}")
    
    def _set_up_actions_with_ball(self, num_team: int):
        idx_init = len(self._actions)
        # Actions With ball:
        for idx in range(num_team):
            self.ACTIONS_WITH_BALL.append("PASS" + str(idx))
        # Add actions to agent actions:
        for act in self.ACTIONS_WITH_BALL:
            self._actions.append(act)
        # Save idxs of actions with ball
        final_idx = len(self._actions)
        self._actions_with_ball_idxs = range(idx_init, final_idx)
    
    def _set_up_actions_without_ball(self):
        idx_init = len(self._actions)
        # Add actions to agent actions:
        for act in self.ACTIONS_WITHOUT_BALL:
            self._actions.append(act)
        # Save idxs of actions with ball
        final_idx = len(self._actions)
        self._actions_without_ball_idxs = range(idx_init, final_idx)
    
    def _step(self, action) -> (int, list):
        status, observation = self.game_interface.step(action)
        self.features.re_calculate_features(
            observation,
            self.game_interface.get_last_player_to_touch_ball())
        return status, observation
    
    def _do_nothing(self, num_rep: int = 1):
        for step in range(num_rep):
            # Game over?
            if not self.game_interface.in_game():
                return
            self._step(action=NOOP)
        return
    
    def _shoot(self) -> (int, list):
        return self._step(action=SHOOT)
    
    def _kick_to_best_angle(self, strength: float = 2.3) -> (int, list):
        angles = []
        player_coord = self.features.agent_coord
        # Calculate angles:
        for goal_pos in self.shoot_possible_coord:
            aux_angles = []
            for op in self.features.opps_coord:
                angle = get_angle(goalie=op, player=player_coord,
                                  point=goal_pos)
                aux_angles.append(angle)
            angles.append(min(aux_angles))
        # Get indexes:
        max_angle = max(angles)
        if max_angle > 5:
            idx = angles.index(max_angle)
            shoot_coord = self.shoot_possible_coord[idx]
            # Action parameters:
            hfo_action = (KICK_TO, shoot_coord[0], shoot_coord[1], strength)
            # Step game:
            return self._step(action=hfo_action)
        else:
            return self._do_nothing()
    
    def get_action_name(self, action_idx) -> str:
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"[Actions] action_idx invalid {action_idx}")
        return self._actions[action_idx]
    
    def get_legal_actions(self):
        if self.features.has_ball():
            return self._actions_with_ball_idxs
        else:
            return self._actions_without_ball_idxs
    
    def get_num_actions(self):
        return self.num_actions
    
    def check_legal_action(self, action_idx: int) -> bool:
        """ Checks if the action is legal """
        if action_idx < 0 or action_idx >= self.num_actions:
            raise ValueError(f"[Actions] action_idx invalid {action_idx}")
        
        if action_idx in self.get_legal_actions():
            return True
        else:
            return False
