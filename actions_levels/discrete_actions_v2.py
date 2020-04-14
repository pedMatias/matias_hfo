from hfo import MOVE_TO, DRIBBLE_TO, KICK_TO, NOOP

from actions_levels import BaseActions


class DiscreteActionsV2:
    """ This class uniforms Move and Dribble actions. It allows agent to only
    have to select between 5 actions, instead of 9 actions
    """
    num_actions = 5
    action_w_ball = ["KICK_TO_GOAL", "DRIBBLE_UP", "DRIBBLE_DOWN",
                     "DRIBBLE_LEFT", "DRIBBLE_RIGHT"]
    action_w_out_ball = ["NOOP", "MOVE_UP", "MOVE_DOWN", "MOVE_LEFT",
                         "MOVE_RIGHT"]
    
    def get_num_actions(self):
        return self.num_actions
    
    def map_action_idx_to_hfo_action(self, agent_pos: tuple, has_ball: bool,
                                     action_idx: int) -> tuple:
        if has_ball:
            action_name = self.action_w_ball[action_idx]
        else:
            action_name = self.action_w_out_ball[action_idx]
        return self.get_action_params(agent_pos, action_name)
    
    def map_action_to_str(self, action_idx: int, has_ball: bool) -> str:
        if has_ball:
            return self.action_w_ball[action_idx]
        else:
            return self.action_w_out_ball[action_idx]
    
    def get_action_params(self, position: tuple, action_name: str) -> tuple:
        x_pos, y_pos = position
        if action_name == "KICK_TO_GOAL":
            return KICK_TO, 0.9, 0, 3
        elif action_name == "NOOP":
            return NOOP
        else:
            if "MOVE" in action_name:
                action = MOVE_TO
            elif "DRIBBLE" in action_name:
                action = DRIBBLE_TO
            else:
                raise ValueError("ACTION NAME is WRONG")
            
            if "UP" in action_name:
                return action, x_pos, - 0.9
            elif "DOWN" in action_name:
                return action, x_pos, 0.9
            elif "LEFT" in action_name:
                return action, -0.8, y_pos
            elif "RIGHT" in action_name:
                return action, 0.8, y_pos
            else:
                raise ValueError("ACTION NAME is WRONG")
