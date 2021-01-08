#!/usr/bin/hfo_env python3
# encoding utf-8
import itertools
import numpy as np
import sys
from collections import defaultdict

import matplotlib
import matplotlib.style
import plotting
import pandas as pd
from hfo import GOAL, OUT_OF_BOUNDS, OUT_OF_TIME, CAPTURED_BY_DEFENSE

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse

class Environment:
    pass

env = Environment()


class QLearningAgent(Agent):
    """
     Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy
     policy while improving following an epsilon-greedy policy;
    :attr learning_rate (float) - Step length taken to update the estimation;
    :attr discount_factor (float) - Discounting Factor for Future Rewards.
        Future rewards are less valuable than current rewards so they must be
        discounted.
    """
    def __init__(self, learning_rate: float = 0.6, discountFactor: float = 0.99,
                 epsilon: float = 1.0, init_vals=0.0):
        super(QLearningAgent, self).__init__()

        # Action value function
        # A nested dictionary that maps
        # state -> (action -> action-value).
        Q = defaultdict(lambda: np.zeros(env.action_space.n))

        # Create an epsilon greedy policy function
        # appropriately for environement_features action space
        policy = self.createEpsilonGreedyPolicy(Q, epsilon, env.action_space.n)

        # For every episode
        for ith_episode in range(num_episodes):

            # Reset the environement_features and pick the first action
            state = env.reset()

            for t in itertools.count():

                # get probabilities of all actions from current state
                action_probabilities = policy(state)

                # choose action according to
                # the probability distribution
                action = np.random.choice(np.arange(
                    len(action_probabilities)),
                    p=action_probabilities)

                # take action and get reward, transit to next state
                next_state, reward, done, _ = env.step(action)

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_lengths[i_episode] = t

                # TD Update
                best_next_action = np.argmax(Q[next_state])
                td_target = reward + discount_factor * Q[next_state][
                    best_next_action]
                td_delta = td_target - Q[state][action]
                Q[state][action] += alpha * td_delta

                # done is True if episode terminated
                if done:
                    break

                state = next_state

        return Q, stats

    def createEpsilonGreedyPolicy(self, Q, epsilon, num_actions):
        """
        Creates an epsilon-greedy policy based
        on a given Q-function and epsilon.

        Returns a function that takes the state
        as an input and returns the probabilities
        for each action in the form of a numpy array
        of length of the action space(set of possible actions).
        """

        def policyFunction(state):
            Action_probabilities = np.ones(num_actions,
                                           dtype=float) * epsilon / num_actions

            best_action = np.argmax(Q[state])
            Action_probabilities[best_action] += (1.0 - epsilon)
            return Action_probabilities

        return policyFunction

    def learn(self):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError

    def to_state_representation(self, state):
        raise NotImplementedError

    def setState(self, state):
        raise NotImplementedError

    def setExperience(self, state, action, reward, status, nextState):
        raise NotImplementedError

    def set_learning_rate(self, learning_rate):
        raise NotImplementedError

    def set_epsilon(self, epsilon):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def compute_hyperparameters(self, num_taken_actions, episodeNumber):
        # TODO implement in the future
        return self.learning_rate, self.epsilon

    def get_reward(self, s: int):
        """
        @param s: game status
                    Possible game statuses:
                    0:  [IN_GAME] Game is currently active
                    1:  [GOAL] A goal has been scored by the offense
                    2:  [CAPTURED_BY_DEFENSE] The defense has captured the ball
                    3:  [OUT_OF_BOUNDS] Ball has gone out of bounds
                    4:  [OUT_OF_TIME] Trial has ended due to time limit
                    5:  [SERVER_DOWN] Server is not alive
        @type s: int
        @return: reward
        @rtype:  int
        """
        if s == GOAL:
            return 1000
        elif s in [CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS]:
            return -1000
        else:
            return -1  # Discount for each time-step


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=0)
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=500)

    args = parser.parse_args()

    # Initialize connection with the HFO server
    hfoEnv = HFOAttackingPlayer(numOpponents=args.numOpponents,
                                numTeammates=args.numTeammates, agentId=args.id)
    hfoEnv.connect_to_server()

    # Initialize a Q-Learning Agent
    agent = QLearningAgent(learning_rate=0.1, discountFactor=0.99, epsilon=1.0)
    num_episodes = args.num_episodes

    # Run training using Q-Learning
    num_taken_actions = 0
    for episode in range(num_episodes):
        status = 0
        observation = hfoEnv.reset()

        while status == 0:
            learning_rate, epsilon = agent.compute_hyperparameters(
                num_taken_actions, episode)
            agent.set_epsilon(epsilon)
            agent.set_learning_rate(learning_rate)

            obsCopy = observation.copy()
            agent.setState(agent.to_state_representation(obsCopy))
            action = agent.act()
            num_taken_actions += 1

            next_observation, reward, done, status = hfoEnv.step(action)
            agent.setExperience(agent.to_state_representation(obsCopy), action,
                                reward, status,
                                agent.to_state_representation(next_observation))
            update = agent.learn()

            observation = next_observation