import numpy as np

from environement_features.discrete_features import DiscreteHighLevelFeatures


class DiscreteFeatures1TeammateV1(DiscreteHighLevelFeatures):
    """
    Features:
        - position: field regions [0,1,2,3,4,5]
        - teammate further : [0, 1]
        - goal opening angle: [0, 1]
        - teammate goal angle: [0, 1]
        - ball_position: [0, 1, 2, 3, 4, 5]
    """
    positions_names = {0: "TOP LEFT", 1: "TOP RIGHT", 2: "MID LEFT",
                       3: "MID RIGHT", 4: "BOTTOM LEFT", 5: "BOTTOM RIGHT"}
    teammate_further = {0: "teammate near goal", 1: "teammate further goal"}
    goal_opening_angle_values = {1: "open angle", 0: "closed_angle"}
    teammate_goal_angle_values = {1: "open angle", 0: "closed_angle"}
    ball_position = {0: "Player Has Ball", 1: "Teammate Has Ball", 2: "Right",
                     3: "Left", 4: "Down", 5: "Up"}
    num_features = 5
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
    
    def _get_ball_position(self) -> int:
        # Agent has ball
        if self._has_ball():
            return 0
        # Teammate has ball
        elif np.linalg.norm(self.teammate_coord - self.ball_coord) <= 0.1:
            return 1
        # Ball direction
        y_diff = abs(self.agent.y_pos - self.agent.ball_y)
        x_diff = abs(self.agent.x_pos - self.agent.ball_x)
        if x_diff >= y_diff and x_diff > 0:
            if self.agent.x_pos < self.agent.ball_x:
                return 2  # RIGHT
            else:
                return 3  # LEFT
        else:
            if self.agent.y_pos < self.agent.ball_y:
                return 4  # DOWN
            else:
                return 5  # UP
            
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
        
    def update_features(self, observation: list):
        """
        Features:
            - position: field regions [0,1,2,3,4,5]
            - teammate further : [0, 1]
            - goal opening angle: [0, 1]
            - teammate goal angle: [0, 1]
            - ball_position: [0, 1, 2, 3, 4, 5]
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
        self.features[4] = self._get_ball_position()
    
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
        size *= len(self.teammate_further)  # teammate relative position
        size *= len(self.goal_opening_angle_values)  # open_angle
        size *= len(self.teammate_goal_angle_values)  # teammates open angle
        size *= len(self.ball_position)
        return size
    
    def get_state_index(self, _=None) -> int:
        idx = 0
        size = 1
        # Agent positions:
        idx += self.features[0] * size
        size *= len(self.positions_names)
        # Teammate relative position:
        idx += self.features[1] * size
        size *= len(self.teammate_further)
        # Agent Open angle:
        idx += self.features[2] * size
        size *= len(self.goal_opening_angle_values)
        # Teammate Open angle:
        idx += self.features[3] * size
        size *= len(self.teammate_goal_angle_values)
        # Ball position
        idx += self.features[4] * size
        size *= len(self.ball_position)
        return int(idx)
    
    def has_ball(self, _=None) -> bool:
        return self.agent.can_kick
    
    def teammate_has_ball(self) -> bool:
        return True if self.features[4] == 1 else False
    
    def teammate_further_from_ball(self) -> bool:
        if np.linalg.norm(self.teammate_coord - self.ball_coord) > \
                np.linalg.norm(self.agent_coord - self.ball_coord):
            return True
        else:
            return False
        
    def teammate_further_from_goal(self) -> bool:
        return True if self.features[1] == 1 else 0
        

