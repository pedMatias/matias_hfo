import numpy as np

from environement_features.discrete_features import DiscreteHighLevelFeatures


class DiscreteFeaturesV2(DiscreteHighLevelFeatures):
    """
    Features:
        - position: field regions [0,1,2,3,4,5]
        - has_ball: [0, 1]
        - goal opening angle: [0, 1]
        - proximity_op: opponent proximity [0, 1]
    """
    positions_names = {0: "TOP LEFT", 1: "TOP RIGHT", 2: "MID LEFT",
                       3: "MID RIGHT", 4: "BOTTOM LEFT", 5: "BOTTOM LEFT"}
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
    
    def update_features(self, observation: list):
        self._encapsulate_data(observation)
        self.features[0] = self._position_finder()
        self.features[1] = self._has_ball()
        self.features[2] = 1 if self.agent.goal_opening_angle > 0.2 else 0
        self.features[3] = 1 if self.agent.proximity_op < -0.4 else 0
    
    def get_position_name(self):
        pos = self.features[0]
        print("Position ID: ", pos)
        return self.positions_names.get(pos)
        
    def get_features(self, _=None):
        return self.features
    
    def get_num_features(self) -> int:
        return self.num_features

    def get_num_states(self):
        """ Returns the total number of possible states """
        size = 6  # positions
        size *= 2  # has ball
        size *= 2  # open_angle
        size *= 2  # opponent proximity
        print("NUM States: ", size)
        return size
    
    def get_state_index(self, _=None) -> int:
        idx = 0
        idx += self.features[0]
        idx += self.features[1] * 6
        idx += self.features[2] * (6 * 2)
        idx += self.features[3] * (6 * 2 * 2)
        return int(idx)
    
    def has_ball(self, _=None) -> bool:
        return self.agent.can_kick
        

