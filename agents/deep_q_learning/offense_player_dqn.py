import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from hfo import HFOEnvironment, HIGH_LEVEL_FEATURE_SET, IN_GAME, GOAL, \
    CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS, MOVE, SHOOT, DRIBBLE, \
    SERVER_DOWN, QUIT

from environement_features.easy_state import EasyState
from agents.deep_q_learning.simple_dqn import DeepQAgent
from agents.deep_q_learning.utils import plotLearning


class Action:
    ids = [SHOOT, DRIBBLE, MOVE]
    names = ["SHOOT", "DRIBBLE", "MOVE"]

    def map_action(self, idx: int) -> int:
        return int(self.ids[idx])

    def get_action_name(self, idx: int) -> str:
        return self.names[idx]

    def get_action_vector(self, can_kick: bool) -> list:
        """ Possible actions [SHOOT, DRIBBLE, MOVE] """
        if can_kick:
            return [1, 1, 0]
        else:
            return [0, 0, 1]


def get_reward(s, can_kick, action):
    reward = 0
    if can_kick and action in [DRIBBLE, SHOOT]:
        reward += 2
    elif not can_kick and action == MOVE:
        reward += 2

    if s == GOAL:
        return 10000 + reward
    elif s in [CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS]:
        return -10000 + reward
    else:
        return -3 + reward


if __name__ == '__main__':
    hfo = HFOEnvironment()
    hfo.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=6000)
    num_teammates = 0
    num_opponents = 1
    lr = 0.0005
    n_games = 1000
    agent = DeepQAgent(gamma=0.99, epsilon=1, alpha=lr, input_dims=4,
                       n_actions=3, mem_size=100000, batch_size=64,
                       epsilon_end=0.0)
    agent.load_model()
    scores = []
    eps_history = []
    for i in range(n_games):
        status = IN_GAME
        score = 0
        observation = hfo.getState()
        env = EasyState(observation, num_team=num_teammates,
                        num_op=num_opponents)
        while status == IN_GAME:
            possible_actions: list = Action().get_action_vector(
                env.agent.can_kick)
            print("> Possible actions: {0}| Epsilon {1}".format(
                possible_actions, agent.epsilon))
            action_idx = agent.choose_action(env.get_representation(),
                                             possible_actions)
            action_id = Action().map_action(action_idx)
            print("> Action: {}".format(hfo.actionToString(action_id)))
            hfo.act(action_id)
            status = hfo.step()
            if status == SERVER_DOWN:
                hfo.act(QUIT)
                break
            reward = get_reward(status, env.agent.can_kick,action_id)
            score += reward
            new_observation = hfo.getState()
            new_env = EasyState(new_observation, num_team=num_teammates,
                                num_op=num_opponents)
            agent.remember(env.get_representation(), action_idx, reward,
                           new_env.get_representation(),
                           done=0 if status == IN_GAME else 1)
            env = new_env

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