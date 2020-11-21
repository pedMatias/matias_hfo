from collections import deque
from typing import List

import numpy as np
from multi_agents.plastic.policy import Policy
from multi_agents.dqn_agent.replay_buffer import Transition


class BehaviourDist:
    INITIAL_VALUE = 1.0
    
    """ Map from probabilities to different policies"""
    def __init__(self, policies: List[Policy], num_features: int,
                 memory_bounded: bool = False, history_len: int = 1):
        # Attributes:
        self._policies = np.array(policies)
        self._team_names = np.array([policy.team_name for policy in policies])
        self._probabilities = np.array([self.INITIAL_VALUE for _ in policies])
        # Memory Bounded Agent:
        if memory_bounded:
            self.memory_bounded = True
            self._transitions_history = deque(maxlen=history_len)
            # n - bounds the maximum allowed loss
            self.n = 1 / history_len
        else:
            self.memory_bounded = False
            self._transitions_history = None
            # n - bounds the maximum allowed loss (Plastic used 0.1)
            self.n = 0.1
        # Max variation of features (used as norm):
        max_arr = np.array([1]*num_features)
        min_arr = np.array([-1]*num_features)
        self.features_max_variation = np.linalg.norm(max_arr-min_arr)
            
    @property
    def team_names(self) -> np.ndarray:
        return self._team_names
    
    def get_policy(self, team_name: str) -> Policy:
        idx = np.where(self._team_names == team_name)[0][0]
        return self._policies[idx]
    
    def get_probability(self, team_name: str) -> float:
        idx = np.where(self._team_names == team_name)[0][0]
        return self._probabilities[idx]
    
    def get_probabilities_dict(self) -> dict:
        aux_dict = {}
        for team_name in self.team_names:
            aux_dict[team_name] = self.get_probability(team_name)
        return aux_dict
        
    def get_best_policy(self) -> Policy:
        max_list = np.where(self._probabilities == np.amax(self._probabilities))
        if len(max_list[0]) > 1:
            policy_idx = np.random.choice(max_list[0])
        else:
            policy_idx = np.argmax(self._probabilities)
        return self._policies[policy_idx]
    
    def _set_policy(self, team_name: str, policy: Policy):
        idx = np.where(self._team_names == team_name)[0][0]
        self._policies[idx] = policy
    
    def _set_probability(self, team_name: str, probability: float):
        idx = np.where(self._team_names == team_name)[0][0]
        self._probabilities[idx] = probability
    
    def _normalize_probabilities(self):
        # Convert values to the sum of 1:
        self._probabilities /= self._probabilities.sum()
    
    def _calc_likelihood_array(self, transition: Transition):
        """
        Returns the likelihood of all the models being the one which the agent
        is interacting with.
        The likelihood is a float value [0, 1]. The highest value, the similar
        it is.
        """
        dist_list = []
        for team in self.team_names:
            policy = self.get_policy(team)
            dist_list.append(policy.model_similarity(transition))
    
        similarity_array = np.array(dist_list)
        # Normalize values:
        likelihood_array = similarity_array / self.features_max_variation
        # Invert values: (the most similar is near 1, else near 0)
        likelihood_array = 1 - likelihood_array
        return likelihood_array
    
    def _calc_new_prob(self, transition: Transition, n: float = 0.1):
        """
        @param transition:
        @param n: bounds the maximum allowed loss (Plastic used 0.1)
        """
        # Get likelihood of each policy:
        likelihood_array = self._calc_likelihood_array(transition)
        # Re-calculate probabilities:
        probabilities = self._probabilities.copy()
        for idx in range(len(probabilities)):
            likelihood = likelihood_array[idx]
            # Belief:
            loss = 1 - likelihood  # loss = 1 −P(a|m, s)
            probabilities[idx] *= (1 - n*loss)  # BehaviorDistr(m)∗=( 1−η.loss)
        self._probabilities = probabilities
    
    def update_beliefs(self, transition: Transition):
        """
        @param transition: Transition
        @return behaviour_dist: updated probability distr
        TODO change the way I calculate similarity
        """
        num_teams = len(self._team_names)
        if self.memory_bounded:
            self._transitions_history.append(transition)
            self._probabilities = np.array([self.INITIAL_VALUE] * num_teams)
            # Update for history len:
            for transition in self._transitions_history:
                self._calc_new_prob(transition, n=self.n)
        else:
            self._calc_new_prob(transition, n=self.n)
        self._normalize_probabilities()

    def select_action(self, s: np.ndarray, legal_actions: list) -> int:
        """
        @param s: the current state
        @param legal_actions: the actions the agent is allowed to use right now
        @return: the best action for the agent to take
        """
        policy: Policy = self.get_best_policy()
        q_predict = policy.dqn.predict(s)[0]
        
        # Set illegal actions to zero:
        for i in range(len(q_predict)):
            if i not in legal_actions:
                q_predict[i] = -2000
        
        # Greedy choice:
        max_list = np.where(q_predict == q_predict.max())
        if len(max_list[0]) > 1:
            action = np.random.choice(max_list[0])
        else:
            action = np.argmax(q_predict)
        return int(action)
