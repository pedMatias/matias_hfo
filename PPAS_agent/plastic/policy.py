import os
import pickle

from multi_agents import config
from multi_agents.dqn_agent.dqn import DQN
from multi_agents.plastic.team_model import TeamModel
from multi_agents.dqn_agent.replay_buffer import Transition


class Policy:
    """ Policy Model """
    
    def __init__(self, team_name, dqn_model: DQN, team_model: TeamModel):
        self.team_name = team_name
        self._dqn = dqn_model
        self._team_model = team_model
    
    @classmethod
    def create(cls, team_name: str, team_dir: str):
        print("[Policy] Creating")
        if not os.path.isdir(team_dir):
            raise ModuleNotFoundError(team_dir)
    
        # Team Model:
        team_model_file = os.path.join(team_dir, config.TEAM_MODEL_FORMAT)
        if not os.path.isfile(team_model_file):
            print(f"[Policy] Creating Team Model: {team_model_file}")
            team_model = TeamModel.set_up_and_save(directory=team_dir)
        else:
            print(f"[Policy] Team Model already created: {team_model_file}")
            team_model = TeamModel.load_model(team_model_file)
        
        # DQN Model:
        dqn_model_file = os.path.join(team_dir, config.DQN_MODEL_FORMAT)
        if not os.path.isfile(dqn_model_file):
            raise FileNotFoundError(f"[Policy] DQN Not found:{dqn_model_file}")
        dqn_model = DQN.load(dqn_model_file)
        return cls(team_name, dqn_model, team_model)
    
    @classmethod
    def load(cls, team_name: str, base_dir: str):
        print(f"[Policy: load] Loading {team_name}...")
        team_dir = os.path.join(base_dir, team_name)
        if not os.path.isdir(team_dir):
            print(f"[Policy: load] Dir not found {team_name};")
            raise NotADirectoryError(team_name)
        
        # DQN Model:
        dqn_model_file = os.path.join(team_dir, config.DQN_MODEL_FORMAT)
        dqn_model = DQN.load(dqn_model_file)
        # Team model:
        team_model_file = os.path.join(team_dir, config.TEAM_MODEL_FORMAT)
        team_model = TeamModel.load_model_from_data(team_model_file)
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
        plastic_file = os.path.join(dir, config.PLASTIC_MODEL_FORMAT)
        with open(plastic_file, 'wb') as f:
            pickle.dump(self, f)
    
    def model_similarity(self, transition: Transition) -> float:
        """
        Returns the likelihood of the model being the one which the agent is
        interacting with.
        The nearest to zero, the similar it is.
        """
        return self.team_model.transition_similarity(transition)
