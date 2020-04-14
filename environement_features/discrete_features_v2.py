import numpy as np

from environement_features.discrete_features import DiscreteHighLevelFeatures


class DiscreteFeaturesV2(DiscreteHighLevelFeatures):
    """
    Features:
        - position: field regions [0,1,2,3,4,5]
        - goal opening angle: [0, 1]
        - proximity_op: opponent proximity [0, 1]
        - ball_position: [0, 1, 2, 3, 4]
    """
    positions_names = {0: "TOP LEFT", 1: "TOP RIGHT", 2: "MID LEFT",
                       3: "MID RIGHT", 4: "BOTTOM LEFT", 5: "BOTTOM LEFT"}
    goal_opening_angle_values = {1: "open angle", 0: "closed_angle"}
    proximity_opponent_values = {1: "opponent close", 0: "opponent far"}
    ball_position = {0: "Player Has Ball", 1: "Up", 2: "Right", 3: "Down",
                     4: "Left"}
    num_features = 4
    features = np.zeros(num_features)

    def _has_ball(self) -> int:
        """
        : return: 1 if agent can kick, else return 0.
        :rtype: int
        """
        return 1 if self.agent.can_kick else 0

    def _position_finder(self):
        """
        :return: Q Table index of agent position.
        Position of agent in terms of quartile block:
            0 == Top Left, 1 == Top right,
            2 == Mid Left, 3 == Mid Right,
            4 == Bottom Left, 5 == Bottom Right
        :rtype: int
        """
        if -1 < self.agent.y_pos < -0.4:
            return 0 if self.agent.x_pos < 0 else 1
        elif -0.4 < self.agent.y_pos < 0.4:
            return 2 if self.agent.x_pos < 0 else 3
        else:  # y in [0.4, 1]
            return 4 if self.agent.x_pos < 0 else 5
    
    def _get_agent_direction(self) -> int:
        """
        : return: integer with agent orientation.
        Orientation of agent in terms of directions:
            0 == Right, 1 == Top, 2 == Left, 3 == Bottom.
        :rtype: int
        """
        orientation = self.agent.orientation
        if orientation in [-0.25, 0.25]:
            return 0
        elif orientation in [0.25, 0.75]:
            return 1
        elif orientation in [-0.75, -0.25]:
            return 2
        else:
            return 3
    
    def _get_ball_direction(self) -> int:
        if self._has_ball():
            return 0
        y_diff = abs(self.agent.y_pos - self.agent.ball_y)
        x_diff = abs(self.agent.x_pos - self.agent.ball_x)
        if x_diff >= y_diff and x_diff > 0:
            if self.agent.x_pos < self.agent.ball_x:
                return 2  # RIGHT
            else:
                return 4  # LEFT
        else:
            if self.agent.y_pos < self.agent.ball_y:
                return 3  # DOWN
            else:
                return 1  # UP
            
    def get_pos_tuple(self) -> tuple:
        """ @return (x axis pos, y axis pos)"""
        return self.agent.x_pos, self.agent.y_pos
        
    def update_features(self, observation: list):
        self._encapsulate_data(observation)
        self.features[0] = self._position_finder()
        self.features[1] = 1 if self.agent.goal_opening_angle > 0.2 else 0
        self.features[2] = 1 if self.agent.proximity_op < -0.4 else 0
        self.features[3] = self._get_ball_direction()
    
    def get_position_name(self):
        pos = self.features[0]
        return self.positions_names.get(pos)
        
    def get_features(self, _=None):
        return self.features
    
    def get_num_features(self) -> int:
        return self.num_features

    def get_num_states(self):
        """ Returns the total number of possible states """
        size = len(self.positions_names)  # positions
        size *= len(self.goal_opening_angle_values)  # open_angle
        size *= len(self.proximity_opponent_values)  # opponent proximity
        size *= len(self.ball_position)
        return size
    
    def get_state_index(self, _=None) -> int:
        idx = 0
        size = 1
        # Agent positions:
        idx += self.features[0] * size
        size *= len(self.positions_names)
        # Open angle:
        idx += self.features[1] * size
        size *= len(self.goal_opening_angle_values)
        # Opponent proximity
        idx += self.features[2] * size
        size *= len(self.proximity_opponent_values)
        # Ball position
        idx += self.features[3] * size
        size *= len(self.ball_position)
        return int(idx)
    
    def has_ball(self, _=None) -> bool:
        return self.agent.can_kick
        

