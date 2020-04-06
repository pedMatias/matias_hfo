from hfo import MOVE_TO, DRIBBLE_TO, KICK_TO

from actions_levels import BaseActions


class DiscreteActions:
    actions = ["MOVE_UP", "MOVE_DOWN", "MOVE_LEFT", "MOVE_RIGHT",
               "KICK_TO_GOAL",
               "DRIBBLE_UP", "DRIBBLE_DOWN", "DRIBBLE_LEFT", "DRIBBLE_RIGHT"]
    
    def get_num_actions(self):
        return len(self.actions)
    
    def map_action_idx_to_hfo_action(self, agent_pos: tuple, action_idx: int)\
            -> tuple:
        action_name = self.actions[action_idx]
        return self.get_action_params(agent_pos, action_name)
    
    def map_action_to_str(self, action_idx: int) -> str:
        return self.actions[action_idx]
    
    def get_action_params(self, position: tuple, action_name: str) -> tuple:
        x_pos, y_pos = position
        if action_name == "KICK_TO_GOAL":
            return KICK_TO, 0.9, 0, 3
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
                return action, x_pos, - 0.9
            elif "LEFT" in action_name:
                return action, -0.8, y_pos
            elif "RIGHT" in action_name:
                return action, 0.8, y_pos
            else:
                raise ValueError("ACTION NAME is WRONG")
