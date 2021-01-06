from typing import List

import numpy as np
from sklearn.neighbors import NearestNeighbors

from agents.plastic_dqn_v1.agent.dqn import DQN
from agents.plastic_dqn_v1.agent.dqn_agent import Transition
from agents.plastic_dqn_v1.plastic.policy import Policy


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