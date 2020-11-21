import numpy as np

from environement_features.base import BaseHighLevelState


HAS_BALL_FEATURE_WEIGHT = 1


class PlasticFeatures(BaseHighLevelState):
    name = "plasticFeatures"
    # (x, y, orientation, goal open, distance to op, has_ball)
    num_default_features = 6
    # (x, y, goal open, distance to op, pass angle, has ball)
    num_teammate_features = 6
    
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
        
        # Coordinates:
        self.ts_coord: np.ndarray = np.array([[0, 0]] * num_team)
        self.opps_coord: np.ndarray = np.array([[0, 0]] * num_op)

        # Teammates:
        self.team_ball_possession = np.array([-1]*self.num_teammates)
        self.teammates_lost_ball = False
        
        self.num_features = self.num_default_features + \
            (num_team * self.num_teammate_features)
        self.features_vect = np.zeros(self.num_features)
    
    @staticmethod
    def near_coords(point1: np.ndarray, point2: np.ndarray,
                    ref: float = 0.15) -> bool:
        dist = abs(np.linalg.norm(point1 - point2))
        return dist <= ref
    
    @property
    def ball_coord(self) -> np.ndarray:
        return np.array([self.agent.ball_x, self.agent.ball_y])
        
    @property
    def agent_coord(self) -> np.ndarray:
        return np.array([self.agent.x_pos, self.agent.y_pos])
    
    def _collect_teammates_coord(self) -> np.ndarray:
        coords = list()
        for teammate in self.teammates:
            coords.append(np.array([teammate.x_pos, teammate.y_pos]))
        return np.array(coords)
    
    def _collect_opponents_coord(self) -> np.ndarray:
        coords = list()
        for opponent in self.opponents:
            coords.append(np.array([opponent.x_pos, opponent.y_pos]))
        return np.array(coords)
    
    def _teammate_has_ball(self, t_id: int, dist_min: float = 0.15) -> bool:
        if self.has_ball():
            return False
        else:
            dd = abs(np.linalg.norm(self.ts_coord[t_id] - self.ball_coord))
            return dd <= dist_min
    
    def _discover_ball_possessions(self, ulp_touch_ball: int):
        """
        Binary vector. If idx 2 is 1.0, means that the teammate 3 has
        the ball
        @param ulp_touch_ball: uniform last player_touch_ball
        """
        t_ball_possession_vector = np.array([-1]*self.num_teammates)
        if self.has_ball():
            return t_ball_possession_vector
        # No teammate touched ball?
        elif ulp_touch_ball not in self.teammates_uniform_numbers:
            return t_ball_possession_vector
        else:
            t_id = self.teammates_uniform_numbers.index(ulp_touch_ball)
            if self._teammate_has_ball(t_id):
                t_ball_possession_vector[t_id] = 1
                return t_ball_possession_vector
            else:  # No teammate has ball:
                return t_ball_possession_vector
    
    def _check_team_lost_ball(self, prev_vector: np.ndarray):
        """ TODO check ball speed """
        if self.has_ball():
            return False
        # Team have ball:
        elif 1 in self.team_ball_possession:
            return False
        # Had ball:
        elif 1 in prev_vector:
            return True
        # No one had ball
        else:
            return False
    
    def update_features(self, observation: list):
        raise Exception()
    
    def re_calculate_features(self, observation: list,
                              last_player_touch_ball_uniform_num: int):
        self._encapsulate_data(observation)
        
        # Auxiliar vars:
        prev_team_ball_possession_vector = self.team_ball_possession.copy()
        
        # Coordinates:
        self.ts_coord: np.ndarray = self._collect_teammates_coord()
        self.opps_coord: np.ndarray = self._collect_opponents_coord()
        
        # Teammates:
        self.team_ball_possession = self._discover_ball_possessions(
            last_player_touch_ball_uniform_num)
        self.teammates_lost_ball = self._check_team_lost_ball(
            prev_vector=prev_team_ball_possession_vector)
        
        # Agent Features (x, y, orientation, goal ang, distance to op,  ball?)
        self.features_vect[0] = self.agent.x_pos
        self.features_vect[1] = self.agent.y_pos
        self.features_vect[2] = self.agent.orientation
        self.features_vect[3] = self.agent.goal_opening_angle
        self.features_vect[4] = self.agent.proximity_op
        self.features_vect[5] = 1 if self.has_ball() else -1
        idx = 6
        for t_id, teammate in enumerate(self.teammates):
            # Teammate features (x, y, goal open, distance to op, pass angle)
            self.features_vect[idx] = teammate.x_pos
            self.features_vect[idx+1] = teammate.y_pos
            self.features_vect[idx+2] = teammate.goal_angle
            self.features_vect[idx+3] = teammate.proximity_op
            self.features_vect[idx+4] = teammate.pass_angle
            self.features_vect[idx+5] = self.team_ball_possession[t_id]
            idx += 6

    def get_features(self, _=None) -> np.ndarray:
        return self.features_vect.copy()
    
    def get_num_features(self) -> int:
        return self.num_features

    def teammates_have_ball(self) -> bool:
        if self.has_ball():
            return False
        elif 1 in self.team_ball_possession:
            return True
        else:
            return False
    
    def get_pos_teammate_with_ball(self) -> np.ndarray:
        if self.has_ball():
            raise ValueError("Function Bad usage")
        elif 1 in self.team_ball_possession:
            team_idx = np.argmax(np.array(self.team_ball_possession))
            return self.ts_coord[team_idx]
        else:
            return None
    
    def opponent_near_position(self, pos: np.ndarray):
        for idx, op_coord in enumerate(self.opps_coord):
            if self.near_coords(pos, op_coord):
                return True
        else:
            return False
    
    def get_nearest_teammate_coord(self) -> (int, np.ndarray):
        near_coord = None
        near_distance = None
        n_team_idx = None
        for idx, t_coord in enumerate(self.ts_coord):
            dist = abs(np.linalg.norm(self.agent_coord - t_coord))
            if near_distance is None or dist < near_distance:
                near_coord = t_coord
                near_distance = dist
                n_team_idx = idx
        # Detect errors
        if n_team_idx is None:
            print("[WARNING - get_nearest_teammate_coord] No teammates!!!")
            n_team_idx, near_coord = (0, (0, 0))
        return n_team_idx, near_coord
    
    def get_nearest_opponent_coord(self) -> (int, np.ndarray):
        near_coord = None
        near_distance = None
        op_idx = None
        for idx, op_coord in enumerate(self.opps_coord):
            dist = abs(np.linalg.norm(self.agent_coord - op_coord))
            if near_distance is None or dist < near_distance:
                near_distance = dist
                near_coord = op_coord
                op_idx = idx
        # Detect errors
        if op_idx is None:
            print("[WARNING - get_nearest_opponent_coord] No opponents!!!")
            op_idx, op_coord = (0, (0, 0))
        return op_idx, near_coord
    
    def near_opponent(self, dist: float = 0.15) -> bool:
        if self.agent.proximity_op < (-1 + dist):
            return True
        else:
            return False
