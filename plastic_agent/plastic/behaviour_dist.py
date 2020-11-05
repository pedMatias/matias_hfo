from collections import deque
from typing import List

import numpy as np
from plastic_agent.plastic.policy import Policy
from plastic_agent.agent.replay_buffer import Transition


class BehaviourDist:
    INITIAL_VALUE = 1.
    
    """ Map from probabilities to different policies"""
    def __init__(self, policies: List[Policy], memory_bounded: bool = False,
                 history_len: int = 1):
        # Attributes:
        self._policies = np.array(policies)
        self._team_names = np.array([policy.team_name for policy in policies])
        self._probabilities = np.array([self.INITIAL_VALUE for _ in policies])
        # Markov agent:
        if memory_bounded:
            self.memory_bounded = True
            self._transitions_history = deque(maxlen=history_len)
        else:
            self.memory_bounded = False
            self._transitions_history = None
            
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
        norm = np.linalg.norm(self._probabilities)
        self._probabilities = self._probabilities / norm
    
    def _calc_likelihood_array(self, transition: Transition):
        """
        Returns the likelihood of all the models being the one which the agent
        is interacting with.
        The likelihood is a float value [0, 1]. The highest value, the similar
        it is.
        """
        similarity_list = []
        for team in self.team_names:
            policy = self.get_policy(team)
            similarity_list.append(policy.model_similarity(transition))
    
        # Normalize values:
        similarity_array = np.array(similarity_list)
        norm = np.linalg.norm(similarity_array)
        likelihood_array = similarity_array / norm
        likelihood_array = 1 - likelihood_array
        return likelihood_array
        
    def update_beliefs(self, transition: Transition, n: int = 0.1,
                       verbose: bool = False):
        """
        @param transition: Transition
        @param n: bounds the maximum allowed loss (Plastic used 0.1)
        @return behaviour_dist: updated probability distr
        TODO change the way I calculate similarity
        """
        if self.memory_bounded:
            self._transitions_history.append(transition)
            # Get likelihood of each policy:
            likelihood_arrays = []
            for t in self._transitions_history:
                likelihood_arrays.append(self._calc_likelihood_array(t))
            # Update belief
            for idx, team in enumerate(self.team_names):
                likelihoods = [l[idx] for l in likelihood_arrays]
                likelihood = sum(likelihoods) / len(likelihoods)
                # Set Belief:
                self._set_probability(team_name=team, probability=likelihood)
            self._normalize_probabilities()
        else:
            # Get likelihood of each policy:
            likelihood_array = self._calc_likelihood_array(transition)
            if verbose:
                old_probabilities = self._probabilities.copy()
                print("*"*20 + " Update Belief " + "*"*20)
                print(":           Teams: ", self.team_names)
                print(":Likelihood array: ", likelihood_array)
            # Update belief
            for idx, team in enumerate(self.team_names):
                probability = self.get_probability(team)
                likelihood = likelihood_array[idx]
                # Belief:
                loss = 1 - likelihood  # loss = 1 −P(a|m, s)
                probability *= (1 - n*loss)  # BehaviorDistr(m)∗ = (1 − η.loss)
                # Set Belief:
                self._set_probability(team_name=team, probability=probability)
            self._normalize_probabilities()
            if verbose:
                print(":Probability var:: ",
                      self._probabilities - old_probabilities)

    def select_action(self, s: np.ndarray) -> int:
        """
        @param s: the current state
        @return: the best action for the agent to take
        """
        policy: Policy = self.get_best_policy()
        q_predict = policy.dqn.predict(s)
        max_list = np.where(q_predict == q_predict.max())
        if len(max_list[0]) > 1:
            action = np.random.choice(max_list[0])
        else:
            action = np.argmax(q_predict)
        return int(action)
