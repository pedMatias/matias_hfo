import argparse

import numpy as np
from hfo import HFOEnvironment, HIGH_LEVEL_FEATURE_SET, IN_GAME, MOVE, SHOOT, DRIBBLE, \
    NOOP, SERVER_DOWN, QUIT

from environement_features.discrete_features import DiscreteHighLevelFeatures
from agents.solo_q_agents.QLearning.qlearner import QLearner
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

    def get_num_actions(self):
        return len(self.ids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numTeammates', type=int, default=0)
    parser.add_argument('--numOpponents', type=int, default=1)
    parser.add_argument('--numEpisodes', type=int, default=1)
    parser.add_argument('--saveFile', type=str, default="q_agent.model")
    args = parser.parse_args()
    num_teammates = args.numTeammates
    num_opponents = args.numOpponents
    n_games = args.numEpisodes
    save_file = args.saveFile
    lr = 0.01
    epsilon = 1
    discount_factor = 0.9
    # Useful Instances:
    hfo = HFOEnvironment()
    hfo.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=6000)
    env = DiscreteHighLevelFeatures(hfo.getState(), num_teammates,
                                    num_opponents)
    actions = Action()
    agent = QLearner(num_states=env.get_num_states(),
                     num_actions=actions.get_num_actions(),
                     epsilon=epsilon,
                     learning_rate=lr,
                     discount_factor=discount_factor,
                     save_file=save_file)
    # Saving lists
    scores = []
    eps_history = []
    for i in range(n_games):
        print("\n<< {}/{} Game >> eps={}".format(i, n_games, agent.epsilon))
        game_status = IN_GAME
        score = 0
        while game_status == IN_GAME:
            action_idx = agent.choose_action(env.get_state_index())
            hfo_action = actions.map_action(action_idx)
            val = hfo.act(hfo_action)
            if val is not None:
                print("Action returned " + str(val))
            game_status = hfo.step()
            if score == 0 and game_status == SERVER_DOWN:
                print("Trying to reconnect")
                hfo.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET,
                                    server_port=6000)
                print("|| Action: {}".format(hfo.actionToString(NOOP)))
                hfo.act(NOOP)
                game_status = hfo.step()
                print("|| New game status: {}".format(
                    hfo.statusToString(game_status)))
                game_status = IN_GAME
            if game_status == SERVER_DOWN:
                hfo.act(QUIT)
                break
            # Reward:
            reward = env.get_reward(game_status, env.agent.can_kick, hfo_action)
            score += reward
            # Environment:
            new_env = DiscreteHighLevelFeatures(hfo.getState(), num_teammates,
                                                num_opponents)
            # Update Q leaning function:
            agent.update_q(env.get_state_index(), action_idx, reward,
                           new_env.get_state_index(),
                           terminal_state=False if game_status == IN_GAME
                                            else True)
            env = new_env
        # Update Epsilon
        agent.update_epsilon()
        # Save metrics
        eps_history.append(agent.epsilon)
        scores.append(score)
        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score: %.2f' % score,
              ' average score %.2f' % avg_score)
        if i % 10 == 0 and i > 0:
            agent.save_q_table()
    # Load Metrics:
    filename = 'game_train.png'
    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)
