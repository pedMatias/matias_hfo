import unittest

from environement_features.discrete_features_v2 import DiscreteFeaturesV2


# OBS = [X position, Y position, Orientation, Ball X, Ball Y, Able to Kick,
#        Goal Center Proximity, Goal Center Angle, Goal Opening Angle,
#        Proximity to Opponent, Opponent X, Opponent Y, Opponent team number,
#        Last action success, Stamina]

OBS_NW_POS = [-0.8, -0.9, -0.14999998, 0.58813035, 0.15183866, 1., -0.76126903,
              -0.18213868, -0.63436294, -0.84415364, 0.76176417, 0.09315622,
              1., 1., 0.63125]
OBS_N_POS = [0, -0.9, -0.14999998, 0.58813035, 0.15183866, 1., -0.76126903,
              -0.18213868, -0.63436294, -0.84415364, 0.76176417, 0.09315622,
              1., 1., 0.63125]
OBS_NE_POS = [0.8, -0.9, -0.14999998, 0.58813035, 0.15183866, 1., -0.76126903,
              -0.18213868, -0.63436294, -0.84415364, 0.76176417, 0.09315622,
              1., 1., 0.63125]
OBS_E_POS = [0.8, 0, -0.14999998, 0.58813035, 0.15183866, 1., -0.76126903,
              -0.18213868, -0.63436294, -0.84415364, 0.76176417, 0.09315622,
              1., 1., 0.63125]
OBS_SE_POS = [0.8, 0.9, -0.14999998, 0.58813035, 0.15183866, 1., -0.76126903,
              -0.18213868, -0.63436294, -0.84415364, 0.76176417, 0.09315622,
              1., 1., 0.63125]
OBS_S_POS = [0, 0.9, -0.14999998, 0.58813035, 0.15183866, 1., -0.76126903,
              -0.18213868, -0.63436294, -0.84415364, 0.76176417, 0.09315622,
              1., 1., 0.63125]
OBS_SW_POS = [-0.8, 0.9, -0.14999998, 0.58813035, 0.15183866, 1., -0.76126903,
              -0.18213868, -0.63436294, -0.84415364, 0.76176417, 0.09315622,
              1., 1., 0.63125]
OBS_W_POS = [-0.8, 0, -0.14999998, 0.58813035, 0.15183866, 1., -0.76126903,
              -0.18213868, -0.63436294, -0.84415364, 0.76176417, 0.09315622,
              1., 1., 0.63125]
OBS_WITH_BALL = [-0.8, 0, -0.14999998, 0.58813035, 0.15183866, 1., -0.76126903,
                 -0.18213868, -0.63436294, -0.84415364, 0.76176417, 0.09315622,
                 1., 1., 0.63125]
OBS_WITHOUT_BALL = [-0.8, 0, -0.14999998, 0.58813035, 0.15183866, -1.,
                    -0.76126903, -0.18213868, -0.63436294, -0.84415364,
                    0.76176417, 0.09315622, 1., 1., 0.63125]
OBS_WITH_OPEN_ANGLE = [-0.8, 0, -0.14999998, 0.58813035, 0.15183866, -1.,
                       -0.76126903, -0.18213868, 0.3, -0.84415364,
                       0.76176417, 0.09315622, 1., 1., 0.63125]
OBS_WITHOUT_OPEN_ANGLE = [-0.8, 0, -0.14999998, 0.58813035, 0.15183866, -1.,
                          -0.76126903, -0.18213868, -0.3, -0.84415364,
                          0.76176417, 0.09315622, 1., 1., 0.63125]
OBS_NEAR_OPPONENT = [-0.8, 0, -0.14999998, 0.58813035, 0.15183866, -1.,
                     -0.76126903, -0.18213868, -0.3, -0.9,
                     0.76176417, 0.09315622, 1., 1., 0.63125]
OBS_FAR_OPPONENT = [-0.8, 0, -0.14999998, 0.58813035, 0.15183866, -1.,
                    -0.76126903, -0.18213868, 0.5, 0.8,
                    0.76176417, 0.09315622, 1., 1., 0.63125]


class TestHighLevelEnvironment(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super(TestHighLevelEnvironment, cls).setUpClass()
        cls.features_manager = DiscreteFeaturesV2(0, 1)

    def test_position_feature(self):
        for obs, pos_id in [(OBS_NW_POS, 0), (OBS_N_POS, 1), (OBS_NE_POS, 1),
                            (OBS_W_POS, 2), (OBS_E_POS, 3),
                            (OBS_SW_POS, 4), (OBS_S_POS, 5), (OBS_SE_POS, 5)]:
            self.features_manager.update_features(obs)
            self.assertEqual(self.features_manager.features[0], pos_id)
    
    def test_has_ball_feature(self):
        # Has ball
        self.features_manager.update_features(OBS_WITH_BALL)
        self.assertTrue(self.features_manager.has_ball())
        # Has no ball
        self.features_manager.update_features(OBS_WITHOUT_BALL)
        self.assertFalse(self.features_manager.has_ball())
    
    def test_open_angle_feature(self):
        # Goal Angle
        self.features_manager.update_features(OBS_WITH_OPEN_ANGLE)
        self.assertEqual(self.features_manager.features[2], 1)
        # Has no ball
        self.features_manager.update_features(OBS_WITHOUT_OPEN_ANGLE)
        self.assertEqual(self.features_manager.features[2], 0)
    
    def test_op_distance_feature(self):
        # Distant Opponent
        self.features_manager.update_features(OBS_FAR_OPPONENT)
        self.assertEqual(self.features_manager.features[3], 0)
        # Has no ball
        self.features_manager.update_features(OBS_NEAR_OPPONENT)
        self.assertEqual(self.features_manager.features[3], 1)




