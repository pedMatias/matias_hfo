import json
import os

import requests

from multi_agents import config
from multi_agents.dqn_agent.dqn import DQN
from multi_agents.plastic.team_model import TeamModel
from multi_agents.dqn_agent.replay_buffer import Transition


class PolicyClient:
    """ Policy Model """
    
    def __init__(self, team_name, dqn_model: DQN, team_model: TeamModel = None,
                 use_webservice: bool = False):
        self.team_name = team_name
        self.use_webservice = use_webservice
        if use_webservice:
            self._dqn = dqn_model
            self._team_model = None
        else:
            assert isinstance(team_model, TeamModel)
            self._dqn = dqn_model
            self._team_model = team_model
    
    @classmethod
    def create(cls, team_name: str, team_dir: str,
               use_webservice: bool = False):
        print("[Policy] Creating")
        if not os.path.isdir(team_dir):
            raise ModuleNotFoundError(team_dir)
    
        # Team Model:
        if use_webservice:
            team_model = None
        else:
            team_model_file = os.path.join(team_dir, config.TEAM_MODEL_FORMAT)
            if not os.path.isfile(team_model_file):
                print(f"[Policy]Creating Team Model: {team_model_file}")
                team_model = TeamModel.set_up_and_save(directory=team_dir)
            else:
                print(f"[Policy]Team Model already created: {team_model_file}")
                team_model = TeamModel.load_model(team_model_file)
        
        # DQN Model:
        dqn_model_file = os.path.join(team_dir, config.DQN_MODEL_FORMAT)
        if not os.path.isfile(dqn_model_file):
            raise FileNotFoundError(f"[Policy] DQN Not found:{dqn_model_file}")
        dqn_model = DQN.load(dqn_model_file)
        return cls(team_name, dqn_model, team_model, use_webservice)
    
    @classmethod
    def load(cls, team_name: str, base_dir: str, use_webservice: bool = False):
        print(f"[Policy: load| use_webservice={use_webservice}] "
              f"Loading {team_name}...")
        team_dir = os.path.join(base_dir, team_name)
        if not os.path.isdir(team_dir):
            print(f"[Policy: load] Dir not found {team_name};")
            raise NotADirectoryError(team_name)
        
        # DQN Model:
        dqn_model_file = os.path.join(team_dir, config.DQN_MODEL_FORMAT)
        dqn_model = DQN.load(dqn_model_file)
        # Team model:
        if use_webservice:
            team_model = None
        else:
            team_model_file = os.path.join(team_dir, config.TEAM_MODEL_FORMAT)
            team_model = TeamModel.load_model_from_data(team_model_file)
        return cls(team_name, dqn_model, team_model, use_webservice)
    
    @property
    def dqn(self) -> DQN:
        return self._dqn
    
    @property
    def team_model(self) -> TeamModel:
        if self._team_model is None:
            raise Exception("Team Model is not trained!")
        else:
            return self._team_model
    
    def model_similarity(self, transition: Transition) -> float:
        """
        Returns the likelihood of the model being the one which the agent is
        interacting with.
        The nearest to zero, the similar it is.
        """
        if self.use_webservice:
            # Not very efficiently:
            data = {"state": transition.obs.tolist(),
                    "next_state": transition.new_obs.tolist()}
            response = requests.post(url=config.URL_POST_SIMILARITY,
                                     json=json.dumps(data))
            return response.json()[self.team_name]
        else:
            return self.team_model.transition_similarity(transition)
