import numpy as np

from environement_features.discrete_features import DiscreteHighLevelFeatures


class ContFeatures1Teammate(DiscreteHighLevelFeatures):
    """
    0: x_pos (float) [-1, 1]
    1: y_pos (float) [-1, 1]
    """
    num_features = 2
    features = np.zeros(num_features)
    # numpy arrays:
    ball_coord = np.array([0, 0])
    agent_coord = np.array([0, 0])
    teammate_coord = np.array([0, 0])
    goalie_coord = np.array([0, 0])
    
    def update_features(self, observation: list):
        self._encapsulate_data(observation)
        # numpy arrays coordenates:
        self.ball_coord = np.array([self.agent.ball_x, self.agent.ball_y])
        self.agent_coord = np.array([self.agent.x_pos, self.agent.y_pos])
        self.teammate_coord = np.array([self.teammates[0].x_pos,
                                       self.teammates[0].y_pos])
        self.goalie_coord = np.array([self.opponents[0].x_pos,
                                      self.opponents[0].y_pos])
        # Features:
        self.features[0] = self.agent.x_pos
        self.features[1] = self.agent.y_pos
        
    def get_pos_tuple(self, round_ndigits: int = -1) -> tuple:
        """ @return (x axis pos, y axis pos)"""
        if round_ndigits >= 0:
            x_pos = round(self.agent.x_pos.item(), round_ndigits)
            x_pos = abs(x_pos) if x_pos == -0.0 else x_pos
            y_pos = round(self.agent.y_pos.item(), round_ndigits)
            y_pos = abs(y_pos) if y_pos == -0.0 else y_pos
            return x_pos, y_pos
        else:
            return self.agent.x_pos, self.agent.y_pos
        
    def get_features(self, _=None) -> np.ndarray:
        return self.features
    
    def get_num_features(self) -> int:
        return self.num_features
    
    def has_ball(self, _=None) -> bool:
        return self.agent.can_kick
        

