import numpy as np

from environement_features.base import BaseHighLevelState


HAS_BALL_FEATURE_WEIGHT = 1


class PlasticFeatures(BaseHighLevelState):
    name = "plasticFeatures"
    # (x, y, orientation, goal open, distance to op, has_ball)
    num_default_features = 6
    # (goal open, distance to op, pass angle)
    num_teammate_features = 3
    
    def __init__(self, num_team: int = None, num_op: int = None):
        """
        @param num_team: number of teammates
        @type num_team: int
        @param num_op: number of opponents
        @type num_op: int
        """
        super().__init__(num_team, num_op)
        self.num_teammates = num_team
        self.num_opponents = num_op
        
        self.num_features = self.num_default_features + \
            (num_team * self.num_teammate_features)
        self.features_vect = np.zeros(self.num_features)
    
    def update_features(self, observation: list):
        self._encapsulate_data(observation)
        # Agent Features (x, y, orientation, goal open, distance to op)
        self.features_vect[0] = self.agent.x_pos
        self.features_vect[1] = self.agent.y_pos
        self.features_vect[2] = self.agent.orientation
        self.features_vect[3] = self.agent.goal_opening_angle
        self.features_vect[4] = self.agent.dist_to_goal
        self.features_vect[5] = 1 if self.has_ball() else -1
        self.features_vect[5] *= HAS_BALL_FEATURE_WEIGHT
        idx = 6
        for teammate in self.teammates:
            # Teammate features (x, y, goal open, distance to op, pass angle)
            self.features_vect[idx] = teammate.goal_angle
            idx += 1
            self.features_vect[idx] = teammate.proximity_op
            idx += 1
            self.features_vect[idx] = teammate.pass_angle
            idx += 1
        
    def get_features(self, _=None) -> np.ndarray:
        return self.features_vect.copy()
    
    def get_num_features(self) -> int:
        return self.num_features
    
    def get_ball_coord(self) -> np.ndarray:
        return np.array([self.agent.ball_x, self.agent.ball_y])

    def get_agent_coord(self) -> np.ndarray:
        return np.array([self.agent.x_pos, self.agent.y_pos])
    
    def get_teammate_coord(self) -> np.ndarray:
        return np.array([self.teammates[0].x_pos, self.teammates[0].y_pos])

    def get_nearest_teammate_coord(self) -> np.ndarray:
        a_coord = self.get_agent_coord()
    
        near_coord = np.array([0, 0])
        near_distance = 100
        for i in range(self.num_teammates):
            t_coord = np.array([self.teammates[i].x_pos,
                                self.teammates[i].y_pos])
            dist = np.linalg.norm(a_coord - t_coord)
            if dist < near_distance:
                near_distance = dist
                near_coord = t_coord
        # Detect errors
        if near_distance == 100:
            print("[WARNING - get_nearest_teammate_coord] No teammates!!!")
        return near_coord

    def get_nearest_opponent_coord(self) -> np.ndarray:
        a_coord = self.get_agent_coord()
    
        near_coord = np.array([0, 0])
        near_distance = 100
        for i in range(self.num_opponents):
            o_coord = np.array([self.opponents[i].x_pos,
                                self.opponents[i].y_pos])
            dist = np.linalg.norm(a_coord - o_coord)
            if dist < near_distance:
                near_distance = dist
                near_coord = o_coord
        # Detect errors
        if near_distance == 100:
            print("[WARNING - get_nearest_opponent_coord] No opponents!!!")
        return near_coord
