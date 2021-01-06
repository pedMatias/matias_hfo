import os
import pickle

from agents.plastic_dqn_v1 import config
from agents.plastic_dqn_v1.agent.dqn import DQN
from agents.plastic_dqn_v1.agent.dqn_agent import Transition
from agents.plastic_dqn_v1.plastic.team_model import TeamModel
from agents.plastic_dqn_v1.agent.replay_buffer import ExperienceBuffer


class Policy:
    """ Policy Model """
    
    def __init__(self, team_name, dqn_model: DQN, team_model: TeamModel):
        self.team_name = team_name
        self._dqn = dqn_model
        self._team_model = team_model
    
    @classmethod
    def create(cls, team_name: str, dir: str):
        print("[Policy] Creating")
        team_dir = os.path.join(dir, team_name)
        try:
            os.mkdir(team_dir)
        except FileExistsError:
            pass
        # DQN Model:
        dqn_model_file = config.DQN_MODEL_FORMAT.format(base_dir=dir,
                                                        team_name=team_name)
        dqn_model = DQN.load(dqn_model_file)
        # Experience:
        replay_buffer_file = config.REPLAY_BUFFER_FORMAT.format(
            base_dir=dir, team_name=team_name)
        replay_buffer = ExperienceBuffer.load(replay_buffer_file)
        print("File ", replay_buffer_file)
        print("Replay Buffer: ", len(replay_buffer.replay_memory),
              replay_buffer.replay_memory[0])
        # Team model:
        team_model = TeamModel.create_model(replay_buffer.to_array())
        team_model_file = config.TEAM_MODEL_FORMAT.format(base_dir=dir,
                                                          team_name=team_name)
        team_model.save_model(team_model_file)
        return cls(team_name, dqn_model, team_model)
    
    @classmethod
    def load(cls, team_name: str, dir: str):
        print("[Policy] Loading")
        team_dir = os.path.join(dir, team_name)
        os.mkdir(team_dir)
        # DQN Model:
        dqn_model_file = config.DQN_MODEL_FORMAT.format(base_dir=dir,
                                                        team_name=team_name)
        dqn_model = DQN.load(dqn_model_file)
        # Team model:
        team_model_file = config.TEAM_MODEL_FORMAT.format(base_dir=dir,
                                                          team_name=team_name)
        team_model = TeamModel.load_model(team_model_file)
        return cls(team_name, dqn_model, team_model)
    
    @property
    def dqn(self) -> DQN:
        return self._dqn
    
    @property
    def team_model(self) -> TeamModel:
        if self._team_model is None:
            raise Exception("Team Model is not trained!")
        else:
            return self._team_model
    
    def save_plastic_model(self, dir: str):
        plastic_file = config.PLASTIC_MODEL_FORMAT.format(
            base_dir=dir,
            team_name=self.team_name)
        with open(plastic_file, 'wb') as f:
            pickle.dump(self, f)
    
    def model_similarity(self, transition: Transition) -> float:
        """
        Returns the likelihood of the model being the one which the agent is
        interacting with
        """
        return self.team_model.similarity(transition)