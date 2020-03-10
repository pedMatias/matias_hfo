import unittest

from environement_features.base import HighLevelState


# OBS = [X position, Y position, Orientation, Ball X, Ball Y, Able to Kick,
#        Goal Center Proximity, Goal Center Angle, Goal Opening Angle,
#        Proximity to Opponent, Opponent X, Opponent Y, Opponent team number,
#        Last action success, Stamina]

OBS_NEAR_GOAL_WITH_BALL = [ 0.55964935, 0.14846742, -0.14999998, 0.58813035,
                            0.15183866, 1., -0.76126903, -0.18213868,
                            -0.63436294, -0.84415364, 0.76176417, 0.09315622,
                            1., 1., 0.63125]

OBS_NEAR_GOAL_WITHOUT_BALL = [ 0.53837264, 0.15852964, -0.1555556, 0.57315946,
                               0.15441191, -1., -0.7434051 , -0.18079472,
                               -0.6602247 , -0.8423122, 0.74148893,
                               0.0990907, 1.,  1.,  0.645  ]

OBS_MEDIUM_WITH_BALL = [ 0.02435672,  0.4169438 , -0.13888884,  0.05458689 ,
                         0.4086709,  1., -0.3044842 , -0.17479873, -0.8768317 ,
                         -0.39781296,  0.740554,  0.07861567,  1.,  1.,
                         0.80499995]

OBS_MEDIUM_WITHOUT_BALL = [-0.04656124,  0.44672418, -0.13333333, -0.01281393,
                           0.43278813, -1., -0.24657279, -0.1726743 ,
                           -0.87627333, -0.29438162,  0.79989016,  0.06135952,
                           1.,  1.,  0.84625006]

OBS_FAR_WITHOUT_BALL = [-0.678823,  0.5445359,  0., -0.66577053, 0.5445359 ,
                        -1., 0.20602918, -0.12860727, -0.8419545,  1.  ,
                        -2.  , -2., -2.  , -1.,  1.  ]

OBS_FAR_WITH_BALL = [-0.6792332,  0.5445584,  0.  , -0.66618073,  0.5445584,
                     1., 0.20631349, -0.12858087, -0.84198624,  1., -2.,
                     -2., -2.,  1.,  1.]

OBS_1TEAMMATE_1OPPONENT = [0.60022545, 0.07074463, 0.4610256, 0.7931396,
                           0.13934898, -1., -0.8182938, -0.11008608,
                           -0.45457488, -0.89646757, -0.6580241, -0.58742595,
                           0., 0.23632705, -0.13154036, 7., 0.7411715,
                           0.0775789, 1., 1., 0.71212494]


class TestHighLevelEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super(TestHighLevelEnvironment, cls).setUpClass()

    def test_enconde_decode_data(self):
        env = HighLevelState(OBS_NEAR_GOAL_WITH_BALL, num_team=0, num_op=1)
        array = env.to_array()
        self.assertTrue(el in OBS_NEAR_GOAL_WITH_BALL for el in array.tolist())
        self.assertEqual(len(OBS_NEAR_GOAL_WITH_BALL), len(array.tolist()))

    def test_enconde_decode_data2(self):
        env = HighLevelState(OBS_1TEAMMATE_1OPPONENT, num_team=1, num_op=1)
        array = env.to_array()
        self.assertTrue(el in OBS_1TEAMMATE_1OPPONENT for el in array.tolist())
        self.assertEqual(len(OBS_1TEAMMATE_1OPPONENT), len(array.tolist()))




