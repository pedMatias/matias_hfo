from hfo import *
import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
import json

from .qlearner import QLearner
from. import state_representer


# Taken from: high_level_sarsa_agent.py in HFO repo
def get_reward(s):
    if s == GOAL:
        return 1000
    elif s in [CAPTURED_BY_DEFENSE, OUT_OF_TIME, OUT_OF_BOUNDS]:
        return -1000
    else:
        return -1   # Discount for each time-step


if __name__ == '__main__':
    hfo = HFOEnvironment()
    hfo.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, server_port=6000)
    FULLSTATE = True

    # Auxiliar structures for metrics:
    scores, episodes = [], []
    registry = {}
    for episode in range(0, args.numEpisodes):
        print(':: Iteration ' + str(episode) + ':')
        status = IN_GAME
        action = None
        state = None
        history = []
        while status == IN_GAME:
            features = hfo.getState()
            # Print off features in a readable manner
            # feature_printer(features, args.numTeammates, args.numOpponents)

            if int(features[5]) != 1:
                history.append((features[0], features[1]))
                if len(history) > 5:
                    history.pop(0)

                # ensures agent does not get stuck for prolonged periods
                if len(history) == 5:
                    if history[0][0] == history[4][0] and history[0][1] == history[4][1]:
                        hfo.act(REORIENT)
                        history = []
                        continue

                hfo.act(MOVE)
            else:
                state, valid_teammates = state_representer.get_representation(features, args.numTeammates)
                # print("Valid Teammates: ", valid_teammates)
                if 0 in valid_teammates:
                    q_learner.set_invalid(state, valid_teammates)

                if action is not None:
                    reward = get_reward(status)
                    # reward_printer(state, action, reward)
                    q_learner.update(state, action, reward)

                action = q_learner.get_action(state, valid_teammates)

                if action == 0:
                    # print("Action Taken: DRIBBLE \n")
                    hfo.act(DRIBBLE)
                elif action == 1:
                    # print("Action Taken: SHOOT \n")
                    hfo.act(SHOOT)
                elif args.numTeammates > 0:
                    # print("Action Taken: PASS -> {0} \n".format(action-2))
                    hfo.act(PASS, features[15 + 6 * (action-2)])
            status = hfo.step()

        reward = None
        if action is not None and state is not None:
            reward = get_reward(status)
            if reward == 0:
                reward = -0.5
            # reward_printer(state, action, reward)
            q_learner.update(state, action, reward)
            q_learner.clear()
            q_learner.save()

        # Save scores:
        scores.append(reward if reward else 0)
        episodes.append(episode)

        if episode % 100 == 0:
            q_learner.save()

        if episode % 1000 == 0:
            q_learner.save(out_file_name + '_iter' + str(episode) + ".npy")
            if len(episodes) >= 10:
                # Create plot:
                rewardfunc = np.polyfit(episodes, scores, 3)
                plt.plot(episodes, scores, 'o')
                trendpoly = np.poly1d(rewardfunc)
                plt.plot(episodes, trendpoly(episodes))
                plt.savefig(GRAPH_FILE + '_ep' + str(episode) + ".png")
                plt.clf()

                # Save Metrics in Json file:
                registry[episode] = {'victories': scores.count(1),
                                     'defeats': scores.count(-1)}
                with open(JSON_FILE + '_ep' + str(episode) + ".json", 'w') as fp:
                    json.dump(registry, fp)

        if status == SERVER_DOWN:
            hfo.act(QUIT)
            break

