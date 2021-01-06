from typing import List

import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree

from agents.plastic_dqn import config
from agents.plastic_dqn.dqn import DQN
from agents.plastic_dqn.dqn_agent import Transition
from agents.plastic_dqn.replay_buffer import ReplayBuffer


class TeamModel:
    """ Saves an KDTree with all the initial states of any transition, and
    also a list with all the Transitions """
    
    def __init__(self, states: list, next_states: list):
        # KDTree:
        self.model: KDTree = KDTree(states)
    
        # Matrix saving all the next states of each state. Each line i
        # corresponds to the next state of state i:
        self.next_states: np.ndarray = np.array(next_states)
    
    @classmethod
    def create_model(cls, data: List[Transition]):
        states = []
        next_states = []
        for transiction in data:
            states.append(transiction.obs)
            next_states.append(transiction.new_obs)
        return cls(states=states, next_states=next_states)
    
    def similarity(self, transition: Transition):
        """
        returns: the similarity between model and transition.
        The nearest to zero, the similar it is.
        """
        state = transition.obs
        next_state = transition.new_obs
        
        nearest_state_idx = self.model.query(state)
        next_nearest_state = self.next_states[nearest_state_idx]
        
        sim = next_state - next_nearest_state
        return abs(np.linalg.norm(sim))
        

class Policy:
    """ Encapsulates a team model. """
    
    def __init__(self, team_name):
        self.team_name = team_name
        
        # DQN
        self._dqn = DQN(num_actions=0, num_features=0,
                        learning_rate=config.LEARNING_RATE)
        self._team_model = None
        self._replay_buffer = ReplayBuffer(
            memory_size=config.REPLAY_MEMORY_SIZE)
    
    @property
    def dqn(self) -> DQN:
        return self._dqn
    
    @property
    def team_model(self) -> TeamModel:
        if self._team_model is None:
            raise Exception("Team Model is not trained!")
        else:
            return self._team_model
    
    def train_team_model(self, data: List[Transition]):
        self._team_model = TeamModel.create_model(data)
    
    def model_similarity(self, transition: Transition) -> float:
        """
        Returns the likelihood of the model being the one which the agent is
        interacting with
        """
        return self.team_model.similarity(transition)
    
    def simulate_teammates_actions(self, state):
        """
        Given a state, predicts a possible set of teammate actions,
        Given their policies
        """
        policies = self.policies(state)
        num_action = 4
        indices = [np.random.choice(range(num_action), p=pi) for pi in
                   policies]
        return indices
    
    def learn_teammate_nn_model(self, data: List[Transition]):
        return NearestNeighbors(n_neighbors=1, algorithm='auto').fit(data)


class BehaviourDist:
    INITIAL_VALUE = 1.
    
    """ Map from probabilities to different policies"""
    def __init__(self, policies: List[Policy]):
        team_names = list()
        policies = list()
        probabilities = list()
        for policy in policies:
            team_names.append(policy.team_name)
            policies.append(policy)
            probabilities.append(self.INITIAL_VALUE)
        # Attributes:
        self.team_names = np.array(team_names)
        self.policies = np.array(policies)
        self.probabilities = np.array(probabilities)
    
    def get_team_names(self):
        return self.team_names
    
    def get_policy(self, team_name: str) -> Policy:
        idx = np.where(self.team_names == team_name)[0][0]
        return self.policies[idx]
    
    def get_probability(self, team_name: str) -> float:
        idx = np.where(self.team_names == team_name)[0][0]
        return self.probabilities[idx]
    
    def set_policy(self, team_name: str, policy: Policy):
        idx = np.where(self.team_names == team_name)[0][0]
        self.policies[idx] = policy
    
    def set_probability(self, team_name: str, probability: float):
        idx = np.where(self.team_names == team_name)[0][0]
        self.probabilities[idx] = probability
    
    def normalize_probabilities(self):
        norm = np.linalg.norm(self.probabilities)
        return self.probabilities / norm
    
    def get_best_policy(self) -> Policy:
        max_list = np.where(self.probabilities == np.amax(self.probabilities))
        if len(max_list[0]) > 1:
            policy_idx = np.random.choice(max_list[0])
        else:
            policy_idx = np.argmax(self.probabilities)
        return self.policies[policy_idx]
        

def learn_about_prior_teammate(teammate_name: str) -> (DQN, NearestNeighbors):
    """
    Plays with the teammates for N timesteps.
    At each timestep collects (s, a, r, s');
    Learns a policy P using DQN
    Learns a nearest neighbors model M
    :@return (P, N)
    """
    policy = Policy(team_name=teammate_name)
    data: List[Transition] = play(train=True, model=policy.dqn)
    # Train Team Model:
    policy.train_team_model(data)
    return policy.dqn, policy.team_model


def update_beliefs(behaviour_dist: BehaviourDist, transition: Transition,
                   n: int) -> BehaviourDist:
    """
    @param behaviour_dist: Map from probabilities to different policies
    @param transition: Transition
    @param n: bounds the maximum allowed loss (Plastic used 0.1)
    @return behaviour_dist: updated probability distr
    TODO change the way I calculate similarity
    """
    team_names = behaviour_dist.get_team_names()
    # Get likelihood of each policy:
    similarity_list = []
    for team in team_names:
        policy = behaviour_dist.get_policy(team)
        similarity_list.append(policy.model_similarity(transition))
    
    # Normalize values:
    similarity_array = np.array(similarity_list)
    norm = np.linalg.norm(similarity_array)
    likelihood_array = norm - similarity_array
    likelihood_array = likelihood_array / norm
    
    # Update belief
    for idx, team in enumerate(team_names):
        probability = behaviour_dist.get_probability(team)
        likelihood = likelihood_array[idx]
        
        # Belief:
        loss = 1 - likelihood  # loss = 1 −P(a|m, s)
        probability *= (1 - n*loss)  # BehaviorDistr(m)∗ = (1 − η.loss)
        
        # Set Belief:
        behaviour_dist.set_probability(team_name=team, probability=probability)
        
    behaviour_dist.normalize_probabilities()
    return behaviour_dist


def select_action(behaviour_distr: BehaviourDist, s: np.ndarray) -> int:
    """
    @param behaviour_distr: probability distr. over possible teammate behaviors
    @param s: the current state
    @return: the best action for the agent to take
    """
    policy: Policy = behaviour_distr.get_best_policy()
    a: int = policy.dqn.predict(s)
    return a


def plastic(directory: str):
    behavior_distr = BehaviourDist.load(dir=directory)
    
    s = None
    n = 0.1  # Value used
    while not terminal(s):
        a = select_action(behavior_distr, s)
        r, new_s = step(a)
        behavior_distr = update_beliefs(behavior_distr, transition, n)