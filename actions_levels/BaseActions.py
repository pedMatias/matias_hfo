from typing import Optional

from hfo import SHOOT, PASS, DRIBBLE, MOVE, GO_TO_BALL, REORIENT, NOOP, QUIT,\
    DRIBBLE_TO, MOVE_TO, KICK_TO


class ActionManager:
    _actions = [SHOOT, PASS, DRIBBLE, MOVE, GO_TO_BALL, REORIENT, NOOP, QUIT,
                MOVE_TO, DRIBBLE_TO, KICK_TO]
    _names = ["SHOOT", "PASS", "DRIBBLE", "MOVE", "GO_TO_BALL", "REORIENT",
              "NOOP", "QUIT", "MOVE_TO", "DRIBBLE_TO", "KICK_TO"]

    @staticmethod
    def valid_action(hfo_action: int, has_ball: bool, params: tuple) -> tuple:
        if hfo_action in [KICK_TO, SHOOT, DRIBBLE_TO, PASS, DRIBBLE] and \
                has_ball:
            return (hfo_action, *params)
        elif hfo_action in [MOVE, MOVE_TO, GO_TO_BALL, DRIBBLE_TO] \
                and not has_ball:
            return (hfo_action, *params)
        elif hfo_action in [REORIENT, NOOP, QUIT]:
            return (hfo_action, *())
        else:
            return (NOOP, *())
    
    def __init__(self, actions: list):
        self.actions_map = {}
        for idx, action in enumerate(actions):
            self.actions_map[idx] = action
    
    def map_action(self, map_idx: int) -> int:
        return int(self.actions_map[map_idx])
    
    def get_num_actions(self) -> int:
        return len(self.actions_map)
    
    def map_action_to_str(self, map_idx: int) -> str:
        hfo_action_id = self.map_action(map_idx)
        action_name = self._names[self._actions.index(hfo_action_id)]
        return action_name
        
    def map_str_to_action(self, action_str):
        id = self._names.index(action_str)
        return self._actions[id]
    
    def get_hfo_action_name(self, hfo_action: int) -> str:
        idx = self._actions.index(hfo_action)
        return self._names[idx]
    
    def get_action_idx_name(self, idx: int) -> str:
        return self._names[idx]

