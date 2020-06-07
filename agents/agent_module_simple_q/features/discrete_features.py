import numpy as np

from environement_features.discrete_features import DiscreteHighLevelFeatures


class DiscreteFeatures1Teammate(DiscreteHighLevelFeatures):
    """
    Features:
        - position: field regions [0,1,2,3,4,5]
        - teammate further : [0, 1]
        - goal opening angle: [0, 1]
        - teammate goal angle: [0, 1]
        - ball_x_pos: [-1, 0, 1]
        - ball_y_pos: [-1, 0, 1]
        - ball_owner: [0, 1, 2]
    """
    positions_names = {0: "TOP LEFT", 1: "TOP RIGHT", 2: "MID LEFT",
                       3: "MID RIGHT", 4: "BOTTOM LEFT", 5: "BOTTOM RIGHT"}
    teammate_further = {0: "teammate near goal", 1: "teammate further goal"}
    goal_opening_angle_values = {1: "open angle", 0: "closed_angle"}
    teammate_goal_angle_values = {1: "open angle", 0: "closed_angle"}
    ball_x_pos = {-1: "Ball Left", 1: "Same x", 2: "Ball Right"}
    ball_y_pos = {-1: "Ball Up", 1: "Same y", 2: "Ball Down"}
    ball_owner = {0: "Player Has Ball", 1: "Teammate Has Ball",
                  2: "No one has ball"}
    num_features = 7
    features = np.zeros(num_features)
    # numpy arrays:
    ball_coord = np.array([0, 0])
    agent_coord = np.array([0, 0])
    teammate_coord = np.array(([0, 0]))

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
    
    def _teammate_further(self) -> int:
        """
        @return: 0 if teammate near goal, 1 otherwise
        @rtype: int
        """
        goal_coord = np.array([1, 0])
        if self.teammate_coord[0] == -2 or self.teammate_coord[1] == -2:
            return 1  # invalid teammate position (out of scope)
        
        team_dist = np.linalg.norm(self.teammate_coord - goal_coord)
        agent_dist = np.linalg.norm(self.agent_coord - goal_coord)
        if team_dist < agent_dist:
            return 0
        else:
            return 1
    
    def _get_ball_owner(self) -> int:
        # Agent has ball
        if self._has_ball():
            return 0
        # Teammate has ball
        elif np.linalg.norm(self.teammate_coord - self.ball_coord) <= 0.1:
            return 1
        # No one has ball:
        else:
            return 2
    
    def _get_ball_x_relative_pos(self) -> int:
        # Ball direction
        x_diff = abs(self.agent.x_pos - self.agent.ball_x)
        if x_diff <= 0.05:
            return 0
        elif self.agent.x_pos > self.agent.ball_x:
            return -1  # LEFT
        else:
            return 1  # RIGHT
    
    def _get_ball_y_relative_pos(self) -> int:
        # Ball direction
        y_diff = abs(self.agent.y_pos - self.agent.ball_y)
        if y_diff <= 0.05:
            return 0
        elif self.agent.y_pos > self.agent.ball_y:
            return -1  # UP
        else:
            return 1  # DOWN
            
    def update_features(self, observation: list):
        """
        Features:
            - position: field regions [0,1,2,3,4,5]
            - teammate further : [0, 1]
            - goal opening angle: [0, 1]
            - teammate goal angle: [0, 1]
            - ball_x_pos: [-1, 0, 1]
            - ball_y_pos: [-1, 0, 1]
            - ball_owner: [0, 1, 2]
        """
        self._encapsulate_data(observation)
        # numpy arrays coordenates:
        self.ball_coord = np.array([self.agent.ball_x, self.agent.ball_y])
        self.agent_coord = np.array([self.agent.x_pos, self.agent.y_pos])
        self.teammate_coord = np.array([self.teammates[0].x_pos,
                                       self.teammates[0].y_pos])
        # features:
        self.features[0] = self._position_finder()
        self.features[1] = self._teammate_further()
        self.features[2] = 1 if abs(self.agent.goal_opening_angle) > 0.2 else 0
        self.features[3] = 1 if abs(self.teammates[0].goal_angle) > 0.2 else 0
        self.features[4] = self._get_ball_x_relative_pos()
        self.features[5] = self._get_ball_y_relative_pos()
        self.features[6] = self._get_ball_owner()
        
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
    
    def get_position_name(self):
        pos = self.features[0]
        return self.positions_names.get(pos)
        
    def get_features(self, _=None) -> np.ndarray:
        return self.features
    
    def get_num_features(self) -> int:
        return self.num_features
    
    def has_ball(self, _=None) -> bool:
        return self.agent.can_kick
    
    def teammate_has_ball(self) -> bool:
        return True if self.features[6] == 1 else False
    
    def teammate_further_from_ball(self) -> bool:
        if np.linalg.norm(self.teammate_coord - self.ball_coord) > \
                np.linalg.norm(self.agent_coord - self.ball_coord):
            return True
        else:
            return False
        
    def teammate_further_from_goal(self) -> bool:
        return True if self.features[1] == 1 else 0
        

