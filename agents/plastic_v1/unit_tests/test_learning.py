import unittest

import numpy as np

from agents.dqn_v1.actions.discrete import DiscreteActions
from agents.dqn_v1.q_agent import QAgent

EPISODE_GOAL_TOP_LEFT = [
    [np.array([1, -1, -1, -1, -1, -1, -1]), 5, 0,
     np.array([-1, 1, -1, -1, -1, -1, -1]), False],
    [np.array([-1, 1, -1, -1, -1, -1, -1]), 3, 0,
     np.array([-1, -1, -1, 1, -1, -1, 1]), False],
    [np.array([-1, -1, -1, 1, -1, -1, 1]), 0, 0,
     np.array([-1, -1, -1, 1, -1, -1, -1]), False],
    [np.array([-1, -1, -1, 1, -1, -1, -1]), 4, 1,
     np.array([-1, -1, -1, 1, -1, -1, -1]), True]
]

EPISODE_FAIL_TOP_LEFT = [
    [np.array([1, -1, -1, -1, -1, -1, -1]), 2, 0,
     np.array([1, -1, -1, -1, -1, -1, -1]), False],
    [np.array([1, -1, -1, -1, -1, -1, -1]), 2, -1,
     np.array([1, -1, -1, -1, -1, -1, -1]), True],
]

EPISODE_GOAL_MID_RIGHT = [
    [np.array([-1, -1, -1, 1, -1, -1, 1]), 0, 0,
     np.array([-1, -1, -1, 1, -1, -1, -1]), False],
    [np.array([-1, -1, -1, 1, -1, -1, -1]), 4, 1,
     np.array([-1, -1, -1, 1, -1, -1, -1]), True]
]

EPISODE_FAIL_MID_RIGHT = [
    [np.array([-1, -1, -1, 1, -1, -1, 1]), 1, 0,
     np.array([-1, -1, -1, 1, -1, -1, -1]), False],
    [np.array([-1, -1, -1, 1, -1, -1, -1]), 4, -1,
     np.array([-1, -1, -1, 1, -1, -1, -1]), True]
]


class TestLearningAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super(TestLearningAgent, cls).setUpClass()
        # Parameters
        # Actions Interface:
        cls.actions = DiscreteActions()
        cls.agent = QAgent(num_features=8,
                           num_actions=cls.actions.get_num_actions(),
                           learning_rate=0.1, discount_factor=0.9, epsilon=0.8)

    def test_fail(self):
        self.agent.store_episode(EPISODE_FAIL_TOP_LEFT)
        # Train:
        # for obs, a, r, new_obs, d, has_ball in EPISODE_FAIL_TOP_LEFT:
        #     print("PREV: st={} -> q={}".format(obs, self.agent.get_qs(obs)))
        self.agent.train(terminal_state=True)
        # for obs, a, r, new_obs, d, has_ball in EPISODE_FAIL_TOP_LEFT:
        #     print("NEW: st={} -> q={}".format(obs, self.agent.get_qs(obs)))
    
    def test_goal_1(self):
        self.agent.store_episode(EPISODE_GOAL_TOP_LEFT)
        # Train:
        # for obs, a, r, new_obs, d, has_ball in EPISODE_FAIL_TOP_LEFT:
        #     print("PREV: st={} -> q={}".format(obs, self.agent.get_qs(obs)))
        self.agent.train(terminal_state=True)
        # for obs, a, r, new_obs, d, has_ball in EPISODE_FAIL_TOP_LEFT:
        #     print("NEW: st={} -> q={}".format(obs, self.agent.get_qs(obs)))
