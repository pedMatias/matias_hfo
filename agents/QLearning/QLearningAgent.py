#!/usr/bin/env python3
# encoding utf-8
import random
from typing import Optional

import numpy as np

from legacy_agents.DiscreteHFO import HFOAttackingPlayer
import argparse


class QLearningAgent:

    def __init__(self, num_actions: int, state_size: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.9,
                 epsilon: float = 0.1, net_file: Optional[str] = None):
        self.epsilon = epsilon
        self.learn_rate = learning_rate
        self.discount = discount_factor
        if net_file:
            self.q_parameters = self._load_parameters(net_file)
            self.replay_buffer = self._load_replay_buffer(net_file)
        else:
            self.q_parameters = np.zeros((state_size, num_actions))
            self.replay_buffer = list()

    def _load_parameters(self, file: Optional[str]):
        raise NotImplementedError

    def _load_replay_buffer(self, file: Optional[str]):
        raise NotImplementedError

    def act(self):
        """ Called at each loop iteration to choose and execute an action.
        Returns:
            action - int
        """
        explore = True if np.random.random() < self.epsilon else False
        if explore:
            random_action = np.random.randint(0, len(self.possible_actions))
            return random_action

        # If multiple equal q-values, pick randomly
        max_list = np.where(self.q_parameters == self.q_parameters.max())[0]
        return random.choice(max_list)

    def learn(self, action: int, reward: int, d: bool):
        """ Called at each loop iteration when the agent is learning. It should
        implement the learning procedure.
        Returns:
            None
        """
        # If game over:
        if d:
            y = reward
        else:
            y = reward + self.discount * np.amax(self.q_parameters)
        new_q_parameters = self.q_parameters.copy()
        new_q_parameters[action] = (1 - self.learn_rate)
        new_q_parameters[action] += self.learn_rate * (
                        reward + self.discount * np.amax(self.q_table[state]))
        self.old_state = state

    def learn(self):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError

    def toStateRepresentation(self, state):
        raise NotImplementedError

    def setState(self, state):
        raise NotImplementedError

    def setExperience(self, state, action, reward, status, nextState):
        raise NotImplementedError

    def setLearningRate(self, learningRate):
        raise NotImplementedError

    def setEpsilon(self, epsilon):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def computeHyperparameters(self, numTakenActions, episodeNumber):
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numEpisodes', type=int, default=500)

    args = parser.parse_args()

    # Initialize connection with the HFO server
    hfoEnv = HFOAttackingPlayer(numOpponents=args.numOpponents,
                                numTeammates=args.numTeammates,
                                agentId=args.id)
    hfoEnv.connect_to_server()

    # Initialize a Q-Learning Agent
    agent = QLearningAgent(learningRate=0.1, discountFactor=0.99, epsilon=1.0)
    numEpisodes = args.numEpisodes

    # Run training using Q-Learning
    numTakenActions = 0
    for episode in range(numEpisodes):
        status = 0
        observation = hfoEnv.reset()

        while status == 0:
            learningRate, epsilon = agent.computeHyperparameters(
                numTakenActions, episode)
            agent.setEpsilon(epsilon)
            agent.setLearningRate(learningRate)

            obsCopy = observation.copy()
            agent.setState(agent.toStateRepresentation(obsCopy))
            action = agent.act()
            numTakenActions += 1

            nextObservation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.toStateRepresentation(obsCopy), action,
                                reward, status,
                                agent.toStateRepresentation(nextObservation))
            update = agent.learn()

            observation = nextObservation
