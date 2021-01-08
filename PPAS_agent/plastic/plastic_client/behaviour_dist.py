import json
import math
from collections import deque
from typing import List

import numpy as np
import requests

from multi_agents.plastic.plastic_client.policy import PolicyClient
from multi_agents.dqn_agent.replay_buffer import Transition
from multi_agents import config


class BehaviourDistClient:
    INITIAL_VALUE = 1.0
    
    """ Map from probabilities to different policies"""
    def __init__(self, policies: List[PolicyClient], num_features: int,
                 agent_type: str, history_len: int = 1,
                 correct_policy: str = None, model_type: str = "stochastic"):
        # Attributes:
        self.num_teams = len(policies)
        self._policies = np.array(policies)
        self._team_names = np.array([policy.team_name for policy in policies])
        self._probabilities = np.array([self.INITIAL_VALUE for _ in policies])
        
        self.agent_type = agent_type
        # Memory Bounded Agent:
        if agent_type == config.AGENT_TYPE_MEMORY_BOUNDED:
            self._transitions_history = deque(maxlen=history_len)
            # n - bounds the maximum allowed loss
            self.n = 1 / history_len
        # Plastic Agent:
        elif agent_type == config.AGENT_TYPE_PLASTIC:
            self._transitions_history = None
            # n - bounds the maximum allowed loss (Plastic used 0.1)
            self.n = 0.1
        # Random Policy:
        elif agent_type == config.AGENT_TYPE_RANDOM_POLICY:
            self._transitions_history = None
            self.n = 0
        # Correct Policy
        else:
            self._transitions_history = None
            self.n = None
            self._probabilities = np.array(
                [1 if t == correct_policy else 0 for t in self._team_names])
            print(self._probabilities)
            print(correct_policy, self._team_names)
            
        # Max variation of features (used as norm):
        max_arr = np.array([1]*num_features)
        min_arr = np.array([-1]*num_features)
        self.features_max_variation = np.linalg.norm(max_arr-min_arr)
        
        # Beliefs mode:
        if model_type in ["stochastic", "adversarial"]:
            self.model_type = model_type
        else:
            raise ValueError("Mode:type. Expected (stochastic, adversarial)")
    
    @property
    def team_names(self) -> np.ndarray:
        return self._team_names
    
    def _is_stochastic(self) -> bool:
        return self.model_type == "stochastic"
    
    def _is_adversarial(self) -> bool:
        return self.model_type == "adversarial"
    
    def _set_policy(self, team_name: str, policy: PolicyClient):
        idx = np.where(self._team_names == team_name)[0][0]
        self._policies[idx] = policy
    
    def _request_similarity(self, transition: Transition) -> dict:
        data = {"state": transition.obs.tolist(),
                "next_state": transition.new_obs.tolist()}
        response = requests.post(url=config.URL_POST_SIMILARITY,
                                 json=json.dumps(data))
        return response.json()
    
    def _normalize_probabilities(self):
        # Convert values to the sum of 1:
        self._probabilities /= self._probabilities.sum()

    def _calc_similarity_array(self, transition: Transition) -> np.ndarray:
        """
        Returns the likelihood of all the models being the one which the agent
        is interacting with.
        The likelihood is a float value [0, 1]. The lowest value, the similar
        it is.
        """
        dist_list = []
        sim_dict = self._request_similarity(transition=transition)
        for team in self.team_names:
            dist_list.append(sim_dict[team])
    
        similarity_array = np.array(dist_list)
        # Normalize values:
        similarity_array = similarity_array / self.features_max_variation
        return similarity_array
    
    def _baysian_update(self, transition: Transition, n: float = 0.1):
        """ polynomial weights algorithm from regret minimization
         
         function UpdateBeliefs(BehDistr, s, a): 15:
            for (π,m) ∈ BehDistr do
                loss = 1 −P(a|m, s)
                BehDistr(m)∗ = (1 − ηloss)
            Normalize BehDistr
            return BehDistr
        """
        # Get similarity between all the available policies:
        similarity_array = self._calc_similarity_array(transition)
        # Invert values: (the most similar is near 1, else near 0)
        likelihood_array = 1 - similarity_array
        
        # Re-calculate probabilities:
        for idx in range(len(self._probabilities)):
            # loss = 1 −P(a|m, s)
            loss = 1 - likelihood_array[idx]
            # BehaviorDistr(m)∗=( 1−η.loss):
            self._probabilities[idx] *= (1 - (n * loss))
    
    def _adversarial_update(self, transition: Transition, n: float = 0.1):
        """ Adversarial Plastic Policy """
        # Get similarity between models. The lowest value, the similar it is:
        similarity_array = self._calc_similarity_array(transition)

        # Re-calculate probabilities:
        for idx, prob in enumerate(self._probabilities):
            # £.di
            var = n * similarity_array[idx]
            # Wi(t+1) = Wi(t).exp(-£.di):
            self._probabilities[idx] = prob * math.exp(-var)
    
    def _calc_new_prob(self, transition: Transition):
        """
        @param transition:
        """
        if self._is_stochastic():
            return self._baysian_update(transition)
        elif self._is_adversarial():
            return self._adversarial_update(transition)
        else:
            raise ValueError()
    
    def _select_stochastic_action(self, s: np.ndarray, legal_actions: list):
        policy: PolicyClient = self.get_best_policy()
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
    
    def _select_adversarial_action(self, s: np.ndarray, legal_actions: list):
        """ Adversarial Plastic Policy """
        # Get Advice vectors (£) for each team:
        advice_vectors = []
        for policy in self._policies:
            q_predict = policy.dqn.predict(s)[0]
            # Add to advice vectors:
            advice_vectors.append(q_predict)
        
        # Get Wt:
        total_probabilities = self._probabilities.sum()
        
        # Calculate actions probabilities:
        num_actions = len(advice_vectors[0])
        act_probs = np.array([-2000] * num_actions)
        for action in legal_actions:
            action_teams_prob = 0
            for t_idx, t_prob in enumerate(self._probabilities):
                action_prob = t_prob * advice_vectors[t_idx][action]
                action_teams_prob += (action_prob / total_probabilities)
            act_probs[action] = action_teams_prob
        
        # Greedy choice:
        max_list = np.where(act_probs == act_probs.max())
        if len(max_list[0]) > 1:
            action = np.random.choice(max_list[0])
        else:
            action = np.argmax(act_probs)
        return int(action)

    def get_probability(self, team_name: str) -> float:
        idx = np.where(self._team_names == team_name)[0][0]
        return self._probabilities[idx]

    def get_probabilities_dict(self) -> dict:
        aux_dict = {}
        for team_name in self.team_names:
            aux_dict[team_name] = self.get_probability(team_name)
        return aux_dict
    
    def update_beliefs(self, transition: Transition):
        """
        @param transition: Transition
        @return behaviour_dist: updated probability distr
        """
        num_teams = len(self._team_names)
        # RANDOM Policy:
        if self.agent_type == config.AGENT_TYPE_RANDOM_POLICY:
            self._probabilities = np.random.random(num_teams)
        # Correct Policy:
        elif self.agent_type == config.AGENT_TYPE_CORRECT_POLICY:
            pass
        # Memory Bounded:
        elif self.agent_type == config.AGENT_TYPE_MEMORY_BOUNDED:
            self._transitions_history.append(transition)
            self._probabilities = np.array([self.INITIAL_VALUE] * num_teams)
            # Update for history len:
            for transition in self._transitions_history:
                self._calc_new_prob(transition)
        # Plastic Agent:
        else:
            self._calc_new_prob(transition)
        self._normalize_probabilities()
    
    def select_action(self, s: np.ndarray, legal_actions: list) -> int:
        """
        @param s: the current state
        @param legal_actions: the actions the agent is allowed to use right now
        @return: the best action for the agent to take
        """
        if self._is_stochastic():
            return self._select_stochastic_action(s, legal_actions)
        elif self._is_adversarial():
            return self._select_adversarial_action(s, legal_actions)
        else:
            raise ValueError()
    
    def get_best_policy(self) -> PolicyClient:
        max_list = np.where(
            self._probabilities == np.amax(self._probabilities))
        if len(max_list[0]) > 1:
            policy_idx = np.random.choice(max_list[0])
        else:
            policy_idx = np.argmax(self._probabilities)
        return self._policies[policy_idx]
    
    def get_best_policy_name(self) -> str:
        max_list = np.where(
            self._probabilities == np.amax(self._probabilities))
        if len(max_list[0]) > 1:
            policy_idx = np.random.choice(max_list[0])
        else:
            policy_idx = np.argmax(self._probabilities)
        team_name = self._team_names[policy_idx]
        return team_name
