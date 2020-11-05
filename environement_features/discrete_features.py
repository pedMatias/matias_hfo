import numpy as np
from hfo import GOAL, CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS, MOVE, \
    SHOOT, DRIBBLE

from environement_features.base import BaseHighLevelState


class DiscreteHighLevelFeatures(BaseHighLevelState):
    num_features = 5

    def _get_direction(self) -> int:
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

    def _has_ball(self) -> int:
        """
        : return: 1 if agent can kick, else return 0.
        :rtype: int
        """
        return 1 if self.agent.able_to_kick else 0

    def _position_finder(self):
        """
        :return: Q Table index of agent position.
        Position of agent in terms of quartile block:
            0 == Top Left, 1 == Top right,
            2 == Mid Left, 3 == Mid Right,
            4 == Bottom Left, 5 == Bottom Right
        :rtype: int
        """
        if self.agent.y_pos in [-1, -0.4]:
            return 0 if self.agent.x_pos < 0 else 1
        elif self.agent.y_pos in [-0.4, 0.4]:
            return 2 if self.agent.x_pos < 0 else 3
        else:  # y in [0.4, 1]
            return 4 if self.agent.x_pos < 0 else 5

    def get_features(self, obs_arr: list):
        """
        Returns a simpler discrete representation of the state
        """
        self._encapsulate_data(obs_arr)
        features = np.zeros(5)

        features[0] = self._position_finder()
        features[1] = self._get_direction()
        features[2] = self._has_ball()
        features[3] = 1 if self.agent.goal_opening_angle > 0.2 else 0
        features[4] = 1 if self.agent.proximity_op > 0.7 else 0
        return features
    
    def get_num_features(self) -> int:
        return self.num_features

    def get_num_states(self):
        """ Returns the total number of possible states """
        size = 6  # positions
        size *= 4  # directions
        size *= 2  # has ball
        size *= 2  # open_angle
        size *= 2  # opponent proximity
        print("NUM States: ", size)
        return size

    def get_state_index(self, observation: list) -> int:
        features = self.get_features(observation)
        idx = 0
        idx += features[0]
        idx += features[1] * 6
        idx += features[2] * (6 * 4)
        idx += features[3] * (6 * 4 * 2)
        idx += features[4] * (6 * 4 * 2 * 2)
        return int(idx)

