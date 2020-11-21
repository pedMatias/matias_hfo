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
        
        def __init__(self, goal_angle, proximity_op, pass_angle, x_pos, y_pos,
                     uniform_num):
            self.goal_angle = goal_angle
            self.proximity_op = proximity_op
            self.pass_angle = pass_angle
            self.x_pos = x_pos
            self.y_pos = y_pos
            self.uniform_num = uniform_num
        
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
        self.length_raw_array = 10 + 6*num_team + 3*num_op + 2
        self.obs_array = np.random.random(self.length_raw_array)
        self.num_teammates = num_team
        self.teammates_uniform_numbers = [-1] * num_team
        self.num_opponents = num_op
        
    def _check_input_dimensions(self, obs_dim):
        """ check if the input size corresponds to expected dimensions """
        if obs_dim == self.length_raw_array:
            return True
        else:
            raise ObservationArrayExpectedDimFail()

    def _remove_outliers(self, obs_arr):
        """ Invalid features are given a value of -2. So we will use previous
        value in order to stabilize training """
        if -2 in obs_arr:
            for idx, val in enumerate(obs_arr):
                if val <= 1.5:
                    print(f"FOUND {idx} with value {val}")
                    obs_arr[idx] = self.obs_array[idx]
        else:
            return obs_arr

    def _get_teammates(self, array: list, num_teammates: int) -> list:
        # Features:
        teammates_goal_opening_angle = array[:num_teammates]
        proximities_to_opponents = array[num_teammates:num_teammates * 2]
        pass_opening_angles = array[num_teammates * 2:num_teammates * 3]
        # Update array:
        array = array[num_teammates*3:]
        # Create Teammates Features modules:
        teammates_dict = dict()
        for t_idx in range(num_teammates):
            aux_array = array[t_idx * 3:]
            x_pos = aux_array[0]
            y_pos = aux_array[1]
            uniform_number = aux_array[2]
            tt = self.Teammate(
                goal_angle=teammates_goal_opening_angle[t_idx],
                proximity_op=proximities_to_opponents[t_idx],
                pass_angle=pass_opening_angles[t_idx],
                x_pos=x_pos, y_pos=y_pos, uniform_num=uniform_number
            )
            teammates_dict[uniform_number] = tt
        # Order teammates by uniform number:
        teammates_list = []
        for t_idx, uniform_num in enumerate(sorted(teammates_dict.keys())):
            self.teammates_uniform_numbers[t_idx] = uniform_num
            teammates_list.append(teammates_dict[uniform_num])
        return teammates_list
    
    def _get_opponents(self, array: np.ndarray, num_op: int) -> list:
        opponents_dict = dict()
        for op in range(0, num_op):
            index = 3 * op
            uni = array[index + 2]
            opponents_dict[uni] = self.Opponent(array[index: index + 3])
        return [opponents_dict[uni] for uni in sorted(opponents_dict.keys())]
        
    def _encapsulate_data(self, obs_arr: list):
        # Check input and normalize values:
        self._check_input_dimensions(len(obs_arr))
        obs_arr = self._remove_outliers(obs_arr)
        # Update observation array:
        self.obs_array = obs_arr
        # Create agent module:
        self.agent = self.Agent(obs_arr)
        # Parse Teammates features:
        self.teammates = self._get_teammates(obs_arr[10:], self.num_teammates)
        idx = 10 + self.num_teammates * 6
        self.opponents = self._get_opponents(obs_arr[idx:], self.num_opponents)
    
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
