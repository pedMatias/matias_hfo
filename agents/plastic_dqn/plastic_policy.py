from agents.plastic_dqn import config
from models.TeamModel import TeamModel
from agents.persuit_plastic.policy.PLASTICPolicy import PLASTICPolicy
from yaaf import mkdir


class PlasticPolicy:
    
    def __init__(self, num_teammates):
        self._num_teammates = num_teammates
        self.total_timesteps = 0
        self.dqn, self.teammates_model = self.setup_models()
    
    @staticmethod
    def setup_dqn(trainable):
        dqn = DQN(trainable, config.LEARNING_RATE, config.DQN_DISCOUNT_FACTOR,
                  ReplayMemoryModel.parse_layer_blueprints(cfg["layers"]),
                  config.REPLAY_MIN_BATCH, config.REPLAY_MEMORY_SIZE)
        return dqn
    
    def reinforce(self, state, joint_actions, reward, next_state, terminal):
        info = {}
        datapoint = state, joint_actions, reward, next_state, terminal
        self.total_timesteps += 1
        info["dqn"] = self.dqn.replay_fit(datapoint)
        info["team model"] = self.team_model.replay_fit(datapoint)
        return info
    
    def save(self, directory):
        prior_dir = f"{directory}/{self.name}"
        mkdir(prior_dir)
        self.dqn.save(prior_dir)
        self.team_model.save(prior_dir)
    
    def load(self, directory):
        prior_dir = f"{directory}/{self.name}"
        self.dqn.load(prior_dir)
        self.team_model.load(prior_dir)
    
    ##################
    # PLASTIC Policy #
    ##################
    
    def setup_models(self):
        self.dqn = self.setup_dqn(trainable=True)
        self.team_model = TeamModel(self._num_teammates, trainable=True)
        return self.dqn, self.team_model
    
    #################
    # PLASTIC Prior #
    #################
    
    def policies(self, state):
        return self.team_model.policies(state)
