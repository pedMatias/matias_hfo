import numpy as np

from environement_features.base import BaseHighLevelState


class PlasticFeatures(BaseHighLevelState):
    num_default_features = 5  # (x, y, orientation, goal open, distance to op)
    num_teammate_features = 5  # (x, y, goal open, distance to op, pass angle)
    
    def __init__(self, num_team: int, num_op: int):
        """
        # TODO Normalize inputs
        @param num_team: number of teammates
        @type num_team: int
        @param num_op: number of opponents
        @type num_op: int
        """
        super().__init__(num_team, num_op)
        self.num_teammates = num_team
        self.num_opponents = num_op
        
        self.num_features = self.num_default_features + \
            num_team * self.num_teammate_features
        self.features_vect = np.zeros(self.num_features)
        
        # Useful structures:
        self.agent_coord = np.zeros(2)
        self.goalie_coord = np.zeros(2)
    
    def _update_coordenates(self):
        self.agent_coord = np.array([self.agent.x_pos, self.agent.y_pos])
        self.goalie_coord = np.array([self.opponents[0].x_pos,
                                      self.opponents[0].y_pos])
    
    def update_features(self, observation: list):
        self._encapsulate_data(observation)
        self._update_coordenates()
        # Agent Features (x, y, orientation, goal open, distance to op)
        self.features_vect[0] = self.agent.x_pos
        self.features_vect[1] = self.agent.y_pos
        self.features_vect[2] = self.agent.orientation
        self.features_vect[3] = self.agent.goal_opening_angle
        self.features_vect[4] = self.agent.dist_to_goal
        idx = 5
        for teammate in self.teammates:
            # Teammate features (x, y, goal open, distance to op, pass angle)
            self.features_vect[idx] = teammate.x_pos
            idx += 1
            self.features_vect[idx] = teammate.y_pos
            idx += 1
            self.features_vect[idx] = teammate.goal_angle
            idx += 1
            self.features_vect[idx] = teammate.proximity_op
            idx += 1
            self.features_vect[idx] = teammate.pass_angle
            idx += 1
        
    def get_features(self, _=None) -> np.ndarray:
        return self.features_vect
    
    def get_num_features(self) -> int:
        return self.num_features
