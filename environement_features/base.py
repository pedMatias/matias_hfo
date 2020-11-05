import numpy as np

from exceptions import ObservationArrayExpectedDimFail


class BaseHighLevelState:
    class Agent:
        x_pos: float = 0
        y_pos: float = 0
        orientation: float = 0
        ball_x: float = 0
        ball_y: float = 0
        proximity_op: float = 0
        goal_opening_angle: float = 0
        able_to_kick: float = 0
        dist_to_goal: float = 0
        goal_center_angle: float = 0
        last_action_succ: float = 0
        stamina: float = 0
        
        def __init__(self, array: list):
            self.x_pos = array[0]
            self.y_pos = array[1]
            self.orientation = array[2]
            self.ball_x = array[3]
            self.ball_y = array[4]
            self.able_to_kick = array[5]  # 1=True
            self.dist_to_goal = array[6]
            self.goal_center_angle = array[7]
            self.goal_opening_angle = array[8]
            self.proximity_op = array[9]
            self.last_action_succ = array[-2]  # 1=True
            self.stamina = array[-1]
        
        def to_array(self):
            return [self.x_pos, self.y_pos, self.orientation, self.ball_x,
                    self.ball_y, self.able_to_kick, self.dist_to_goal,
                    self.goal_center_angle, self.goal_opening_angle,
                    self.proximity_op, self.last_action_succ,
                    self.stamina]
        
        def len(self):
            return len(self.to_array())
    
    class Teammate:
        goal_angle: float = 0
        proximity_op: float = 0
        pass_angle: float = 0
        x_pos: float = 0
        y_pos: float = 0
        uniform_num: int = 0
        
        def __init__(self, array: list):
            self.goal_angle = array[0]
            self.proximity_op = array[1]
            self.pass_angle = array[2]
            self.x_pos = array[3]
            self.y_pos = array[4]
            self.uniform_num = array[5]
        
        def to_array(self):
            return [self.goal_angle, self.proximity_op, self.pass_angle,
                    self.x_pos, self.y_pos, self.uniform_num]
        
        def len(self):
            return len(self.to_array())
    
    class Opponent:
        x_pos: float = 0
        y_pos: float = 0
        uniform_num: int = 0
        
        def __init__(self, array: list):
            self.x_pos = array[0]
            self.y_pos = array[1]
            self.uniform_num = array[2]
        
        def to_array(self):
            return [self.x_pos, self.y_pos, self.uniform_num]
        
        def len(self):
            return len(self.to_array())
        
    def __init__(self, num_team: int = 0, num_op: int = 0):
        """
        # TODO Normalize inputs
        @param num_team: number of teammates
        @type num_team: int
        @param num_op: number of opponents
        @type num_op: int
        """
        self.num_teammates = num_team
        self.num_opponents = num_op
        
    def _check_input_dimensions(self, obs_dim, num_team, num_op):
        """ check if the input size corresponds to expected dimensions """
        expected_dim = 10 + num_team * 6 + num_op * 3 + 2
        if obs_dim == expected_dim:
            return True
        else:
            raise ObservationArrayExpectedDimFail()

    def _get_teammates(self, array: list, num_team: int) -> list:
        teammates = list()
        for t in range(0, num_team):
            index = 6 * t
            teammates.append(self.Teammate(array[index: index + 6]))
        return teammates

    def _get_opponents(self, array: np.ndarray, num_op: int) -> list:
        opponents = []
        for op in range(0, num_op):
            index = 3 * op
            opponents.append(self.Opponent(array[index: index + 3]))
        return opponents

    def _encapsulate_data(self, obs_arr: list):
        self._check_input_dimensions(len(obs_arr), self.num_teammates,
                                     self.num_opponents)
        self.agent = self.Agent(obs_arr)
        self.teammates = self._get_teammates(obs_arr[10:], self.num_teammates)
        idx = 10 + self.num_teammates * 6
        self.opponents = self._get_opponents(obs_arr[idx:], self.num_opponents)

    def to_array(self) -> np.ndarray:
        agent_array = self.agent.to_array()
        teammates_array = []
        for teammate in self.teammates:
            teammates_array += teammate.to_array()
        opponents_array = []
        for opponent in self.opponents:
            opponents_array += opponent.to_array()
        return np.array(agent_array + teammates_array + opponents_array)
    
    def has_ball(self) -> bool:
        if self.agent.able_to_kick == 1:
            return True
        else:
            return False
    
    def get_pos_tuple(self, round_ndigits: int = -1) -> tuple:
        """ @return (x axis pos, y axis pos)"""
        x_pos = self.agent.x_pos.item()
        y_pos = self.agent.y_pos.item()
        if round_ndigits >= 0:
            x_pos = round(x_pos, round_ndigits)
            x_pos = abs(x_pos) if x_pos == -0.0 else x_pos
            y_pos = round(y_pos, round_ndigits)
            y_pos = abs(y_pos) if y_pos == -0.0 else y_pos
            return x_pos, y_pos
        else:
            return x_pos, y_pos

    def update_features(self, observation: list):
        return self._encapsulate_data(observation)
