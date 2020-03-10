import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from hfo import HFOEnvironment, HIGH_LEVEL_FEATURE_SET, IN_GAME, GOAL, \
    CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS, MOVE, SHOOT, DRIBBLE, NOOP

from agents.deep_q_learning.high_level_state import HighLevelState
from agents.deep_q_learning.simple_dqn import DeepQAgent
from agents.deep_q_learning.utils import plotLearning

ACTIONS = [MOVE, SHOOT, DRIBBLE, NOOP]


def get_reward(s):
    if s == GOAL:
        return 1000
    elif s in [CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS]:
        return -1000
    else:
        return -1   # Discount for each time-step


def map_action(action: int) -> int:
    return int(ACTIONS[action])


def get_possible_actions_vector(env: HighLevelState) -> list:
    if env.state.can_kick:
        possible_actions = [SHOOT, DRIBBLE, NOOP]
    else:
        possible_actions = [MOVE, NOOP]
    return [i for i in range(len(ACTIONS)) if ACTIONS[i] in possible_actions]


if __name__ == '__main__':
    hfo = HFOEnvironment()
    hfo.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=6000)
    num_teammates = 0
    num_opponents = 1
    lr = 0.0005
    n_games = 500
    agent = DeepQAgent(gamma=0.99, epsilon=0.0, alpha=lr, input_dims=13,
                       n_actions=4, mem_size=1000000, batch_size=64,
                       epsilon_end=0.0)
    scores = []
    eps_history = []
    for i in range(n_games):
        status = IN_GAME
        score = 0
        observation = hfo.getState()
        env = HighLevelState(observation, num_team=num_teammates,
                               num_op=num_opponents)
        while status == IN_GAME:
            possible_actions: list = get_possible_actions_vector(env)
            action_ind = agent.choose_action(env.to_array(), possible_actions)
            action = map_action(action_ind)
            hfo.act(action)
            status = hfo.step()
            reward = get_reward(status)
            score += reward
            new_observation = hfo.getState()
            new_env = HighLevelState(new_observation,
                                     num_team=num_teammates,
                                     num_op=num_opponents)
            agent.remember(env.to_array(), action_ind, reward, new_env.to_array(),
                           done=0 if status == IN_GAME else 1)
            observation = new_observation
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score: %.2f' % score,
              ' average score %.2f' % avg_score)

        if i % 10 == 0 and i > 0:
            agent.save_model()

    filename = 'game_test.png'

    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)