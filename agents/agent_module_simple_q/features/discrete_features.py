import numpy as np

from environement_features.discrete_features import DiscreteHighLevelFeatures


class DiscreteFeatures1Teammate(DiscreteHighLevelFeatures):
    # Features
    agent_x_pos: float  # [-1, 1]
    agent_y_pos: float  # [-1, 1]
    teammate_x_pos: float  # [-1, 1]
    teammate_x_pos: float  # [-1, 1]
    ball_x_pos: float  # [-1, 1]
    ball_y_pos: float  # [-1, 1]
    agent_has_ball: float = {-1.: "no ball", 1.: "has ball"}
    teammate_has_ball: float = {-1.: "no ball", 1.: "has ball"}
    goal_opening_angle_values = {1: "open angle", -1: "closed_angle"}
    teammate_goal_angle_values = {1: "open angle", -1: "closed_angle"}
    
    # features vector
    num_features = 10
    features = np.zeros(num_features)
    
    # numpy arrays:
    ball_coord = np.array([0, 0])
    agent_coord = np.array([0, 0])
    teammate_coord = np.array(([0, 0]))
    
    def update_features(self, observation: list):
        self._encapsulate_data(observation)
        # features:
        self.features[0] = self.agent.x_pos
        self.features[1] = self.agent.y_pos
        self.features[2] = self.teammates[0].x_pos
        self.features[3] = self.teammates[0].y_pos
        self.features[4] = self.agent.ball_x
        self.features[5] = self.agent.ball_y
        self.features[6] = 1 if self.has_ball() else -1
        self.features[7] = 1 if self.check_teammate_has_ball() else -1
        self.features[8] = 1 if abs(self.agent.goal_opening_angle) > 0.2 else 0
        self.features[9] = 1 if abs(self.teammates[0].goal_angle) > 0.2 else 0
            
    def get_features(self, _=None) -> np.ndarray:
        return self.features
        
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

    def has_ball(self, _=None) -> bool:
        return self.agent.can_kick
    
    def check_teammate_has_ball(self) -> bool:
        distance_to_ball = np.linalg.norm(
            np.array([self.teammates[0].x_pos, self.teammates[0].y_pos]) -
            np.array([self.agent.ball_x, self.agent.ball_y]))
        if distance_to_ball <= 0.05:
            return True
        else:
            return False
    
    def get_num_features(self) -> int:
        return self.num_features
        

