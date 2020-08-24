from abc import ABC, abstractmethod

import numpy as np

from agents.plastic_dqn import config

from environment.PursuitState import PursuitState
from environment.utils import pursuit_datapoint
from yaaf.policies import deterministic_policy
from yaaf.agents import Agent


class PlasticAgent(Agent, ABC):
    """ Plastic Agent - Has n possible team models m, identifies or
    learns the current one. """

    def __init__(self, name: str, num_teammates: int, num_opponents: int,
                 learn_team: bool = True, verbose: bool = False):

        super().__init__(name)

        self.num_teammates = num_teammates
        self.num_opponents = num_opponents
        self.num_agents = self.num_teammates + self.num_opponents + 1

        self.learning_team = learn_team
        if self.learning_team:
            self.priors = [self.setup_learning_prior()]
            self.learning_prior = self.priors[0]
        else:
            self.priors = []

        self.belief_distribution = np.zeros((len(self.priors),)) + 1 / len(self.priors)

        assert np.sum(self.belief_distribution) == 1.0, "Something went wrong initializing the beliefs..."

        self.eta = config.ETA

        self.verbose = verbose

    def update_beliefs(self, previous_observation: np.ndarray,
                       next_observation: np.ndarray):
        # TODO P(a|m, s) for each m in beliefs
        raise NotImplementedError()

        # Compute beliefs given new data
        for i in range(len(self.belief_distribution)):
            loss = 1 - P_a_m_s[i]
            self.belief_distribution[i] *= 1 - self.eta * loss
            self.belief_distribution[i] = float("{0:.5f}".format(self.belief_distribution[i]))

        # Normalize for distribution
        self.belief_distribution = self.belief_distribution / self.belief_distribution.sum()
        return self.belief_distribution

    def most_likely_model(self):
        """
        Given the belief distribution, picks a random model
        The higher a model belief, the higher the probability of it being picked
        """
        choice = self.belief_distribution.argmax()
        if self.verbose:
            print(f"Belief Dist.: {self.belief_distribution}")
            print("Most Likely Team: ", self.priors[choice].name)
        return self.priors[choice]

    def policy(self, curr_observation: np.ndarray):
        features: np.ndarray = np.array()  # TODO parse features
        # TODO action = self.select_action_according_to_model(features, self.most_likely_model())
        return deterministic_policy(action, num_actions=4)
    
    def reinforce(self, state, reward, next_state, terminal):
        info = {}
        datapoint = state, reward, next_state, terminal
        self.total_timesteps += 1
        info["dqn"] = self.dqn.replay_fit(datapoint)
        info["team model"] = self.team_model.replay_fit(datapoint)
        return info

    def _reinforce(self, timestep):
        info = {}
        state, reward, next_state, terminal = pursuit_datapoint(timestep, self.world_size)
        beliefs = self.update_beliefs(state, next_state)
        info["belief distribution"] = {}
        for i, m in enumerate(self.priors):
            info["belief distribution"][m.name] = beliefs[i]
        info["learning prior"] = self.learning_prior.reinforce(state, reward, next_state, terminal)
        return info

    def save(self, directory):
        np.save(f"{directory}/beliefs.npy", self.belief_distribution)
        self.learning_prior.save(directory)

    def save_learning_prior(self, directory, name, clear=True):
        if not self.learning_prior:
            print("WARN: Not learning any team", flush=True)
            return
        self.learning_prior.name = name
        self.learning_prior.save(directory)
        if clear:
            del self.learning_prior
            self.learning_prior = self.setup_learning_prior()
            self.priors[-1] = self.learning_prior

    def load_learnt_prior(self, directory, team_name):
        prior = self._load_prior_team(directory, team_name)
        self.priors = [prior] + self.priors
        del self.belief_distribution
        self.belief_distribution = np.zeros((len(self.priors),)) + 1 / len(self.priors)
        assert np.sum(self.belief_distribution) == 1.0

    def load(self, directory):
        self.learning_prior.load(directory)
        self.belief_distribution = np.load(f"{directory}/beliefs.npy")

    def select_action_according_to_model(self, state, most_likely_model):
            # TODO parse features;
            # TODO predict q values;
            # TODO Select action greedy;
            action = 0
            return action

    def setup_learning_prior(self):
        return LearningPLASTICPolicy(self.num_teammates)

    def _load_prior_team(self, directory, name):
        return LearntPLASTICPolicy(directory, name, self.num_teammates)