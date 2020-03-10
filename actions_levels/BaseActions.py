from hfo import SHOOT, PASS, DRIBBLE, MOVE, GO_TO_BALL, REORIENT, NOOP, QUIT


class ActionManager:
    _actions = [SHOOT, PASS, DRIBBLE, MOVE, GO_TO_BALL, REORIENT, NOOP, QUIT]
    _names = ["SHOOT", "PASS", "DRIBBLE", "MOVE", "GO_TO_BALL", "REORIENT",
             "NOOP", "QUIT"]

    @staticmethod
    def valid_action(hfo_action: int, has_ball: bool):
        if hfo_action in [SHOOT, PASS, DRIBBLE] and has_ball:
            return hfo_action
        elif hfo_action in [MOVE, GO_TO_BALL] and not has_ball:
            return hfo_action
        elif hfo_action in [REORIENT, NOOP, QUIT]:
            return hfo_action
        else:
            return NOOP
    
    def __init__(self, actions: list):
        self.actions_map = {}
        for idx, action in enumerate(actions):
            self.actions_map[idx] = action
        print("Action map: ", self.actions_map)
    
    def map_action(self, map_idx: int) -> int:
        return int(self.actions_map[map_idx])
    
    def get_num_actions(self) -> int:
        print("NUm actions: ", len(self.actions_map))
        return len(self.actions_map)
    
    def map_str_to_action(self, action_str):
        id = self._names.index(action_str)
        return self._actions[id]
    
    def get_hfo_action_name(self, hfo_action: int) -> str:
        idx = self._actions.index(hfo_action)
        return self._names[idx]
    
    def get_action_idx_name(self, idx: int) -> str:
        return self._names[idx]

