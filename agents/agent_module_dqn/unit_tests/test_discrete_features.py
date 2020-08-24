import unittest

import numpy as np

from agents.agent_module_dqn.deep_agent import DQNAgent
from actions_levels.action_module import DiscreteActionsModule
from agents.agent_module_dqn.features.discrete_features import \
    DiscreteFeatures1Teammate

"""
Features:
    - position: field regions [0,1,2,3,4,5]
    - teammate further from goal : [0, 1]
    - goal opening angle: [0, 1]
    - teammate goal angle: [0, 1]
    - ball_x_pos: [-1, 0, 1]
    - ball_y_pos: [-1, 0, 1]
    - ball_owner: [0, 1, 2]
ACTIONS:
    - with ball:
        action_w_ball = ["KICK_TO_GOAL", "PASS","LONG_DRIBBLE_UP",
            "LONG_DRIBBLE_DOWN", "LONG_DRIBBLE_LEFT", "LONG_DRIBBLE_RIGHT",
            "SHORT_DRIBBLE_UP", "SHORT_DRIBBLE_DOWN", "SHORT_DRIBBLE_LEFT",
            "SHORT_DRIBBLE_RIGHT"]
    action_w_out_ball = ["NOOP", "NOOP", "LONG_MOVE_UP", "LONG_MOVE_DOWN",
        "LONG_MOVE_LEFT", "LONG_MOVE_RIGHT", "SHORT_MOVE_UP", "SHORT_MOVE_DOWN",
        "SHORT_MOVE_LEFT", "SHORT_MOVE_RIGHT"]
"""


# _NOTE_: -100 means any value
# 0:pos | 1:t.further | 2:open.angle | 3:team.op.a | 4:b.x | 5:b.y | 6:b.owner
# OBSERVATION -> EXPECTED_ACTIONS
WITH_BALL = {
    # 0:pos| 1:t.furt| 2:open.angle| 3:team.op.a| 4:b.x| 5:b.y| 6:b.owner
    # TOP LEFT:
    (0, 0, 0, 0, 0, 0, 0): ["PASS", "DOWN", "RIGHT"],
    (0, 0, 0, 1, 0, 0, 0): ["PASS", "DOWN", "RIGHT"],
    (0, 0, 1, 0, 0, 0, 0): ["PASS", "KICK", "DOWN", "RIGHT"],
    (0, 0, 1, 1, 0, 0, 0): ["PASS", "DOWN", "RIGHT"],
    (0, 1, 0, 0, 0, 0, 0): ["DOWN", "RIGHT"],
    (0, 1, 0, 1, 0, 0, 0): ["DOWN", "RIGHT"],
    (0, 1, 1, 0, 0, 0, 0): ["KICK", "DOWN", "RIGHT"],
    (0, 1, 1, 1, 0, 0, 0): ["DOWN", "RIGHT"],
    
    # TOP RIGHT:
    (1, 0, 0, 0, 0, 0, 0): ["PASS", "DOWN"],
    (1, 0, 0, 1, 0, 0, 0): ["PASS", "DOWN"],
    (1, 0, 1, 0, 0, 0, 0): ["PASS", "KICK", "DOWN"],
    (1, 0, 1, 1, 0, 0, 0): ["PASS", "DOWN"],
    (1, 1, 0, 0, 0, 0, 0): ["DOWN"],
    (1, 1, 0, 1, 0, 0, 0): ["DOWN"],
    (1, 1, 1, 0, 0, 0, 0): ["KICK", "DOWN"],
    (1, 1, 1, 1, 0, 0, 0): ["DOWN"],
    
    # MID LEFT:
    (2, 0, 0, 0, 0, 0, 0): ["PASS", "RIGHT"],
    (2, 0, 0, 1, 0, 0, 0): ["PASS", "RIGHT"],
    (2, 0, 1, 0, 0, 0, 0): ["PASS", "KICK", "RIGHT"],
    (2, 0, 1, 1, 0, 0, 0): ["PASS", "RIGHT"],
    (2, 1, 0, 0, 0, 0, 0): ["RIGHT"],
    (2, 1, 0, 1, 0, 0, 0): ["RIGHT"],
    (2, 1, 1, 0, 0, 0, 0): ["KICK", "RIGHT"],
    (2, 1, 1, 1, 0, 0, 0): ["RIGHT"],
    
    # MID RIGHT:
    (3, 0, 0, 0, 0, 0, 0): ["PASS", "RIGHT", "KICK"],
    (3, 0, 0, 1, 0, 0, 0): ["PASS", "RIGHT", "KICK"],
    (3, 0, 1, 0, 0, 0, 0): ["PASS", "RIGHT", "KICK"],
    (3, 0, 1, 1, 0, 0, 0): ["PASS", "RIGHT", "KICK"],
    (3, 1, 0, 0, 0, 0, 0): ["RIGHT", "KICK"],
    (3, 1, 0, 1, 0, 0, 0): ["RIGHT", "KICK"],
    (3, 1, 1, 0, 0, 0, 0): ["RIGHT", "KICK"],
    (3, 1, 1, 1, 0, 0, 0): ["RIGHT", "KICK"],
    
    # BOT LEFT:
    (4, 0, 0, 0, 0, 0, 0): ["PASS", "UP", "RIGHT"],
    (4, 0, 0, 1, 0, 0, 0): ["PASS", "UP", "RIGHT"],
    (4, 0, 1, 0, 0, 0, 0): ["PASS", "KICK", "UP", "RIGHT"],
    (4, 0, 1, 1, 0, 0, 0): ["PASS", "UP", "RIGHT"],
    (4, 1, 0, 0, 0, 0, 0): ["UP", "RIGHT"],
    (4, 1, 0, 1, 0, 0, 0): ["UP", "RIGHT"],
    (4, 1, 1, 0, 0, 0, 0): ["KICK", "UP", "RIGHT"],
    (4, 1, 1, 1, 0, 0, 0): ["UP", "RIGHT"],
    
    # BOT RIGHT:
    (5, 0, 0, 0, 0, 0, 0): ["PASS", "UP"],
    (5, 0, 0, 1, 0, 0, 0): ["PASS", "UP"],
    (5, 0, 1, 0, 0, 0, 0): ["PASS", "KICK", "UP"],
    (5, 0, 1, 1, 0, 0, 0): ["PASS", "UP"],
    (5, 1, 0, 0, 0, 0, 0): ["UP"],
    (5, 1, 0, 1, 0, 0, 0): ["UP"],
    (5, 1, 1, 0, 0, 0, 0): ["KICK", "UP"],
    (5, 1, 1, 1, 0, 0, 0): ["UP"]
}

TEAMMATE_HAS_BALL = {
    # 0:pos| 1:t.furt| 2:open.angle| 3:team.op.a| 4:b.x| 5:b.y| 6:b.owner
    # TOP LEFT:
    (0, 0, 0, 0, 0, 0, 1): ["DOWN", "RIGHT"],
    (0, 0, 0, 1, 0, 0, 1): ["DOWN", "RIGHT"],
    (0, 0, 1, 0, 0, 0, 1): ["DOWN", "RIGHT"],
    (0, 0, 1, 1, 0, 0, 1): ["DOWN", "RIGHT"],
    (0, 1, 0, 0, 0, 0, 1): ["DOWN", "RIGHT"],
    (0, 1, 0, 1, 0, 0, 1): ["DOWN", "RIGHT"],
    (0, 1, 1, 0, 0, 0, 1): ["DOWN", "RIGHT"],
    (0, 1, 1, 1, 0, 0, 1): ["DOWN", "RIGHT"],
    
    # TOP RIGHT:
    (1, 0, 0, 0, 0, 0, 1): ["NOOP", "DOWN", "LEFT"],
    (1, 0, 0, 1, 0, 0, 1): ["NOOP", "DOWN", "LEFT"],
    (1, 0, 1, 0, 0, 0, 1): ["NOOP", "DOWN", "LEFT"],
    (1, 0, 1, 1, 0, 0, 1): ["NOOP", "DOWN", "LEFT"],
    (1, 1, 0, 0, 0, 0, 1): ["NOOP", "DOWN", "LEFT"],
    (1, 1, 0, 1, 0, 0, 1): ["NOOP", "DOWN", "LEFT"],
    (1, 1, 1, 0, 0, 0, 1): ["NOOP", "DOWN", "LEFT"],
    (1, 1, 1, 1, 0, 0, 1): ["NOOP", "DOWN", "LEFT"],
    
    # MID LEFT:
    (2, 0, 0, 0, 0, 0, 1): ["UP", "RIGHT", "DOWN"],
    (2, 0, 0, 1, 0, 0, 1): ["UP", "RIGHT", "DOWN"],
    (2, 0, 1, 0, 0, 0, 1): ["UP", "RIGHT", "DOWN"],
    (2, 0, 1, 1, 0, 0, 1): ["UP", "RIGHT", "DOWN"],
    (2, 1, 0, 0, 0, 0, 1): ["UP", "RIGHT", "DOWN"],
    (2, 1, 0, 1, 0, 0, 1): ["UP", "RIGHT", "DOWN"],
    (2, 1, 1, 0, 0, 0, 1): ["UP", "RIGHT", "DOWN"],
    (2, 1, 1, 1, 0, 0, 1): ["UP", "RIGHT", "DOWN"],
    
    # MID RIGHT:
    (3, 0, 0, 0, 0, 0, 1): ["UP", "RIGHT", "DOWN", "LEFT", "NOOP"],
    (3, 0, 0, 1, 0, 0, 1): ["UP", "RIGHT", "DOWN", "LEFT", "NOOP"],
    (3, 0, 1, 0, 0, 0, 1): ["UP", "RIGHT", "DOWN", "LEFT", "NOOP"],
    (3, 0, 1, 1, 0, 0, 1): ["UP", "RIGHT", "DOWN", "LEFT", "NOOP"],
    (3, 1, 0, 0, 0, 0, 1): ["UP", "RIGHT", "DOWN", "LEFT", "NOOP"],
    (3, 1, 0, 1, 0, 0, 1): ["UP", "RIGHT", "DOWN", "LEFT", "NOOP"],
    (3, 1, 1, 0, 0, 0, 1): ["UP", "RIGHT", "DOWN", "LEFT", "NOOP"],
    (3, 1, 1, 1, 0, 0, 1): ["UP", "RIGHT", "DOWN", "LEFT", "NOOP"],
    
    # BOT LEFT:
    (4, 0, 0, 0, 0, 0, 1): ["UP", "RIGHT", "NOOP"],
    (4, 0, 0, 1, 0, 0, 1): ["UP", "RIGHT", "NOOP"],
    (4, 0, 1, 0, 0, 0, 1): ["UP", "RIGHT", "NOOP"],
    (4, 0, 1, 1, 0, 0, 1): ["UP", "RIGHT", "NOOP"],
    (4, 1, 0, 0, 0, 0, 1): ["UP", "RIGHT", "NOOP"],
    (4, 1, 0, 1, 0, 0, 1): ["UP", "RIGHT", "NOOP"],
    (4, 1, 1, 0, 0, 0, 1): ["UP", "RIGHT", "NOOP"],
    (4, 1, 1, 1, 0, 0, 1): ["UP", "RIGHT", "NOOP"],
    
    # BOT RIGHT:
    (5, 0, 0, 0, 0, 0, 1): ["UP", "RIGHT", "LEFT", "NOOP"],
    (5, 0, 0, 1, 0, 0, 1): ["UP", "RIGHT", "LEFT", "NOOP"],
    (5, 0, 1, 0, 0, 0, 1): ["UP", "RIGHT", "LEFT", "NOOP"],
    (5, 0, 1, 1, 0, 0, 1): ["UP", "RIGHT", "LEFT", "NOOP"],
    (5, 1, 0, 0, 0, 0, 1): ["UP", "RIGHT", "LEFT", "NOOP"],
    (5, 1, 0, 1, 0, 0, 1): ["UP", "RIGHT", "LEFT", "NOOP"],
    (5, 1, 1, 0, 0, 0, 1): ["UP", "RIGHT", "LEFT", "NOOP"],
    (5, 1, 1, 1, 0, 0, 1): ["UP", "RIGHT", "LEFT", "NOOP"]
}

NO_ONE_HAS_BALL = {
    # 4:b.x| 5:b.y| 6:b.owner
    (-1, -1, 2): ["LEFT", "UP"],
    (-1, 0, 2): ["LEFT"],
    (-1, 1, 2): ["LEFT", "DOWN"],
    (0, -1, 2): ["UP"],
    (0, 0, 2): ["NOOP"],
    (0, 1, 2): ["DOWN"],
    (1, -1, 2): ["RIGHT", "UP"],
    (1, 0, 2): ["RIGHT"],
    (1, 1, 2): ["RIGHT", "DOWN"],
}


class TestHighLevelEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super(TestHighLevelEnvironment, cls).setUpClass()
        # Parameters
        cls.num_opponents = 1
        cls.num_teammates = 1
        cls.model_file = "/home/matias/Desktop/HFO/matias_hfo/data/new_1ep_2020-05-23_19:15:00/agent_model"
        # Instances:
        cls.actions = DiscreteActionsModule()
        cls.features = DiscreteFeatures1Teammate(num_op=cls.num_opponents,
                                                 num_team=cls.num_teammates)
        cls.agent = DQNAgent(cls.features.get_num_features(),
                             cls.actions.get_num_actions())
        # Set up agent:
        cls.agent.epsilon = 0
        cls.agent.load_model(cls.model_file)

    def test_has_ball(self):
        def check_action(obs: list) -> bool:
            arr = np.array(obs)
            has_ball = True if arr[0] == 0 else False
            action_idx = self.agent.exploit_actions(arr)
            action_name = self.actions.map_action_to_str(action_idx, has_ball)
            if obs not in WITH_BALL.keys():
                raise("Expected observation {}".format(obs))
            for act_base in WITH_BALL[obs]:
                if act_base in action_name:
                    return True
            print(
                "Wrong action!!  {0}->Selected Action={1}->Expected Actions={2}".
                format(obs, action_name, WITH_BALL[obs]))
            return False
        num_sts = 0
        num_oks = 0
        # Feature 0: Position
        for f0 in [0, 1, 2, 3, 4, 5]:
            # Feature 1: Teammate Further
            for f1 in [0, 1]:
                # Feature 2: Open Angle
                for f2 in [0, 1]:
                    # Feature 3: Teammate Open Angle
                    for f3 in [0, 1]:
                        # Feature 4: Ball x relative pos
                        for f4 in [0]:
                            # Feature 5: Ball y relative pos
                            for f5 in [0]:
                                # Feature 6: Ball owner
                                for f6 in [0]:
                                    obs = (f0, f1, f2, f3, f4, f5, f6)
                                    num_oks += 1 if check_action(obs) else 0
                                    num_sts += 1
        print("[Test HAS BALL] {}% accuracy".format((num_oks*100)/num_sts))
        
    def test_teammate_has_ball(self):
        def check_action(obs: list) -> bool:
            arr = np.array(obs)
            has_ball = True if arr[0] == 0 else False
            action_idx = self.agent.exploit_actions(arr)
            action_name = self.actions.map_action_to_str(action_idx, has_ball)
            if obs not in TEAMMATE_HAS_BALL.keys():
                raise("Expected observation {}".format(obs))
            for act_base in TEAMMATE_HAS_BALL[obs]:
                if act_base in action_name:
                    return True
            print(
                "Wrong action!!  {0}->Selected Action={1}->Expected Actions={2}".
                format(obs, action_name, TEAMMATE_HAS_BALL[obs]))
            return False
        num_sts = 0
        num_oks = 0
        # Feature 0: Position
        for f0 in [0, 1, 2, 3, 4, 5]:
            # Feature 1: Teammate Further
            for f1 in [0, 1]:
                # Feature 2: Open Angle
                for f2 in [0, 1]:
                    # Feature 3: Teammate Open Angle
                    for f3 in [0, 1]:
                        # Feature 4: Ball x relative pos
                        for f4 in [0]:
                            # Feature 5: Ball y relative pos
                            for f5 in [0]:
                                # Feature 6: Ball owner
                                for f6 in [1]:
                                    obs = (f0, f1, f2, f3, f4, f5, f6)
                                    num_oks += 1 if check_action(obs) else 0
                                    num_sts += 1
        print("[Test TEAMMATE HAS BALL] {}% accuracy".format(
            (num_oks * 100) / num_sts))
    
    def test_no_ball(self):
        def check_action(obs: list) -> bool:
            arr = np.array(obs)
            has_ball = True if arr[0] == 0 else False
            action_idx = self.agent.exploit_actions(arr)
            action_name = self.actions.map_action_to_str(action_idx, has_ball)
            # Interesting vector part:
            interest_obs = obs[4:]
            if interest_obs not in NO_ONE_HAS_BALL.keys():
                raise("Extected obs {}".format(obs))
            for act_base in NO_ONE_HAS_BALL[interest_obs]:
                if act_base in action_name:
                    return True
            print(
                "Wrong action!!  {0}->Selected Action={1}->Expected Actions={2}".
                format(obs, action_name, NO_ONE_HAS_BALL[interest_obs]))
            return False
        num_sts = 0
        num_oks = 0
        # Feature 0: Position
        for f0 in [0, 1, 2, 3, 4, 5]:
            # Feature 1: Teammate Further
            for f1 in [0, 1]:
                # Feature 2: Open Angle
                for f2 in [0, 1]:
                    # Feature 3: Teammate Open Angle
                    for f3 in [0, 1]:
                        # Feature 4: Ball x relative pos
                        for f4 in [-1, 0, 1]:
                            # Feature 5: Ball y relative pos
                            for f5 in [-1, 0, 1]:
                                # Feature 6: Ball owner
                                for f6 in [2]:
                                    obs = (f0, f1, f2, f3, f4, f5, f6)
                                    num_oks += 1 if check_action(obs) else 0
                                    num_sts += 1
        print("[Test NO ONE HAS BALL] {}% accuracy".format(
            (num_oks * 100) / num_sts))
                                    
        




