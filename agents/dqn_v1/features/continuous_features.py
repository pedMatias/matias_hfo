import numpy as np

from environement_features.discrete_features import DiscreteHighLevelFeatures


class DiscFeatures1Teammate(DiscreteHighLevelFeatures):
    
    # positions_names = {0: "TOP LEFT", 1: "TOP RIGHT", 2: "MID LEFT",
    #                    3: "MID RIGHT", 4: "BOTTOM LEFT", 5: "BOTTOM RIGHT"}
    # teammate_further = {0: "teammate near goal", 1: "teammate further goal"}
    # goal_opening_angle_values = {1: "open angle", 0: "closed_angle"}
    # teammate_goal_angle_values = {1: "open angle", 0: "closed_angle"}
    # ball_x_pos = {-1: "Ball Left", 1: "Same x", 2: "Ball Right"}
    # ball_y_pos = {-1: "Ball Up", 1: "Same y", 2: "Ball Down"}
    # ball_owner = {0: "Player Has Ball", 1: "Teammate Has Ball",
    #               2: "No one has ball"}
    num_features = 6
    features = np.zeros(num_features)
    # numpy arrays:
    ball_coord = np.array([0, 0])
    agent_coord = np.array([0, 0])
    teammate_coord = np.array([0, 0])
    goalie_coord = np.array([0, 0])

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
        self.goalie_coord = np.array([self.opponents[0].x_pos,
                                      self.opponents[0].y_pos])
        # Features:
        # 0-5: Agent Position:
        for i in range(6):  # set positions to "false":
            self.features[i] = -1
        curr_agent_pos = self._position_finder()
        self.features[curr_agent_pos] = 1  # set current position:
        # 6: Agent has open angle to goal?
        #self.features[6] = 1 if abs(self.agent.goal_opening_angle) > 0.2 else 0
        # 7: Agent has ball?
        # self.features[7] = 1 if self.has_ball() else -1
        # 6: Teammate further from goal?
        ## self.features[6] = 1 if self._teammate_further() else -1
        # 8: Teammate has open angle to goal?
        ## self.features[8] = 1 if abs(self.teammates[0].goal_angle) > 0.2  else -1
        # 9: Ball position relative to the agent (x axis):
        ## self.features[9] = self._get_ball_x_relative_pos()
        # 10: Ball position relative to the agent (y axis):
        ## self.features[10] = self._get_ball_y_relative_pos()
        
        ## self.features[11] = self._get_ball_owner()
        
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
        

