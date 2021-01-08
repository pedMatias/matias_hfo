#!/usr/bin/hfo_env python3
# encoding utf-8
import json
import os
import random

import argparse

from hfo import OUT_OF_TIME, SERVER_DOWN

from agents.base.hfo_attacking_player import HFOAttackingPlayer
from environement_features.discrete_features_v2 import DiscreteFeaturesV2
from environement_features.reward_functions import basic_reward
from actions_levels.discrete_actions_v5 import DiscreteActionsV5
from utils.aux_functions import q_table_variation
from agents.solo_q_agents.q_agent_v6 import mkdir, save_model
from agents.solo_q_agents.q_agent_v6 import QLearningAgentV6

ORIGIN_POSITIONS = {"TOP LEFT": (-0.5, -0.7), "TOP RIGHT": (0.4, -0.7),
                    "MID LEFT": (-0.5, 0.0), "MID RIGHT": (0.4, 0.0),
                    "BOTTOM LEFT": (-0.5, 0.7), "BOTTOM RIGHT": (0.4, 0.7)}

DRIBBLE_SHORT = 10  # MOVES
DRIBBLE_LONG = 20  # MOVES

SHORT_KICK_SPEED = 1.5
LONG_KICK_SPEED = 2.6


class NoActionPlayedError(Exception):
    pass


class ServerDownError(Exception):
    pass


def go_to_origin_position(game_interface: HFOAttackingPlayer,
                          features: DiscreteFeaturesV2,
                          actions: DiscreteActionsV5,
                          random_start: bool = True):
    if random_start:
        pos_name, origin_pos = random.choice(list(ORIGIN_POSITIONS.items()))
    else:
        pos_name = "Fixed start"
        origin_pos = features.get_pos_tuple()
    # print("\nMoving to starting point: {0}".format(pos_name))
    pos = features.get_pos_tuple(round_ndigits=1)
    while origin_pos != pos:
        has_ball = features.has_ball()
        hfo_action: tuple = actions.dribble_to_pos(origin_pos)
        status, observation = game_interface.step(hfo_action, has_ball)
        features.update_features(observation)
        pos = features.get_pos_tuple(round_ndigits=1)


def execute_action(action_params: tuple, repetitions:int,
                   game_interface: HFOAttackingPlayer, has_ball: bool):
    rep_counter_aux = 0
    observation = []
    while game_interface.in_game() and rep_counter_aux < repetitions:
        status, observation = game_interface.step(action_params, has_ball)
        rep_counter_aux += 1
    return game_interface.get_game_status(), observation
        

def test(num_episodes: int, game_interface: HFOAttackingPlayer,
         features: DiscreteFeaturesV2, agent: QLearningAgentV6,
         actions: DiscreteActionsV5, reward_funct) -> float:
    """
    @param num_episodes: number of episodes to run
    @param game_interface: game interface, that manages interactions
    between both;
    @param features: features interface, from the observation array, gets the
    main features for the agent;
    @param agent: learning agent;
    @param actions: actions interface;
    @param reward_funct: reward function used
    @return: (float) the average reward
    """
    # Run training using Q-Learning
    sum_score = 0
    for ep in range(num_episodes):
        # Check if server still up:
        if game_interface.hfo.step() == SERVER_DOWN:
            print("Server is down while testing; episode={}".format(ep))
            break
        # Go to origin position:
        features.update_features(game_interface.get_state())
        go_to_origin_position(game_interface=game_interface,
                              features=features, actions=actions)
        # Test loop:
        debug_counter = 0  # TODO remove
        while game_interface.in_game():
            # Update environment features:
            curr_state_id = features.get_state_index()
            has_ball = features.has_ball()

            # Act:
            debug_counter += 1
            action_idx = agent.exploit_actions(curr_state_id)
            hfo_action_params, num_rep = \
                actions.map_action_idx_to_hfo_action(
                    agent_pos=features.get_pos_tuple(), has_ball=has_ball,
                    action_idx=action_idx)
            
            # Step:
            status, observation = execute_action(
                action_params=hfo_action_params, repetitions=num_rep,
                has_ball=has_ball, game_interface=game_interface)
            
            # update features:
            reward = reward_funct(status)
            features.update_features(observation)
            sum_score += reward
        
        if status == OUT_OF_TIME:
            if debug_counter < 5:
                raise NoActionPlayedError("agent was only able to choose {}"
                                          .format(debug_counter))
        # Game Reset
        game_interface.reset()
    print("<<TEST>> AVR reward = ", sum_score / (ep+1))
    return sum_score / num_episodes


def train(num_train_episodes: int, num_total_train_ep: int,
          game_interface: HFOAttackingPlayer, features: DiscreteFeaturesV2,
          agent: QLearningAgentV6, actions: DiscreteActionsV5, reward_funct):
    """
    @param num_train_episodes: number of episodes to train in this iteration
    @param num_total_train_ep: number total of episodes to train
    @param game_interface: game interface, that manages interactions
    between both;
    @param features: features interface, from the observation array, gets
    the main features for the agent;
    @param agent: learning agent;
    @param actions: actions interface;
    @param reward_funct: reward function used
    @return: (QLearningAgentV5) the agent
    """
    sum_score = 0
    sum_epsilons = 0
    agent.counter_explorations = 0
    agent.counter_exploitations = 0
    for ep in range(num_train_episodes):
        # Check if server still up:
        # if game_interface.hfo.step() == SERVER_DOWN:
        #     raise ServerDownError("training; episode={}".format(ep))
        # Go to origin position:
        features.update_features(game_interface.get_state())
        go_to_origin_position(game_interface=game_interface,
                              features=features, actions=actions)
        # Start learning loop
        debug_counter = 0  # TODO remove
        while game_interface.in_game():
            # Update environment features:
            curr_state_id = features.get_state_index()
            has_ball = features.has_ball()
            
            # Act:
            debug_counter += 1
            action_idx = agent.act(curr_state_id)
            hfo_action_params, num_rep =\
                actions.map_action_idx_to_hfo_action(
                    agent_pos=features.get_pos_tuple(), has_ball=has_ball,
                    action_idx=action_idx)

            # Step:
            status, observation = execute_action(
                action_params=hfo_action_params, repetitions=num_rep,
                has_ball=has_ball, game_interface=game_interface)
            
            # Update environment features:
            reward = reward_funct(status)
            sum_score += reward
            features.update_features(observation)
            new_state_id = features.get_state_index()
            agent.store_ep(state_idx=curr_state_id, action_idx=action_idx,
                           reward=reward, next_state_idx=new_state_id,
                           has_ball=has_ball,
                           done=not game_interface.in_game())
        if status == OUT_OF_TIME:
            if debug_counter < 5:
                raise NoActionPlayedError("agent was only able to choose {}".
                                          format(debug_counter))
        agent.learn_buffer()
        agent.update_hyper_parameters(num_total_episodes=num_total_train_ep)
        sum_epsilons += agent.epsilon
        # Game Reset
        game_interface.reset()
    print("<<TRAIN>> AVR reward = ", sum_score / num_train_episodes)
    print("<<TRAIN>> %Explorations={}% ".
          format(
        round((agent.counter_explorations /
               (agent.counter_exploitations + agent.counter_explorations)), 4)
              * 100))


def export_metrics(trained_eps: list, rewards: list, epsilons: list,
                   q_variation: list, save_dir: str):
    """ Saves metrics in Json file"""
    data = {"trained_eps": trained_eps, "epsilons": epsilons,
            "q_table_variation": q_variation, "reward": rewards}
    file_path = os.path.join(save_dir, "metrics.json")
    with open(file_path, 'w+') as fp:
        json.dump(data, fp)


def check_if_q_table_stayed_the_same(qtable1, qtable2):
    q_variation = q_table_variation(qtable1, qtable2)
    if q_variation != 0:
        raise Exception("Q Learning changed after test", q_variation)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_train_ep', type=int, default=1000)
    parser.add_argument('--num_test_ep', type=int, default=0)
    parser.add_argument('--num_repetitions', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default=None)
    
    args = parser.parse_args()
    agent_id = args.id
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_train_ep = args.num_train_ep
    num_test_ep = args.num_test_ep
    num_repetitions = args.num_repetitions
    num_episodes = (num_train_ep + num_test_ep) * num_repetitions
    # Directory
    save_dir = args.save_dir or mkdir(num_episodes, num_op, extra_note="oldEps")
    
    print("Starting Training - id={}; num_opponents={}; num_teammates={}; "
          "num_episodes={};".format(agent_id, num_op, num_team, num_episodes))
    # Initialize connection with the HFO server
    hfo_interface = HFOAttackingPlayer(agent_id=agent_id,
                                       num_opponents=num_op,
                                       num_teammates=num_team)
    hfo_interface.connect_to_server()
    
    # Agent set-up
    reward_function = basic_reward
    features_manager = DiscreteFeaturesV2(num_team, num_op)
    actions_manager = DiscreteActionsV5()
    agent = QLearningAgentV6(num_states=features_manager.get_num_states(),
                             num_actions=actions_manager.get_num_actions(),
                             learning_rate=0.1, discount_factor=0.9,
                             epsilon=0.8)

    # Test one first time without previous train:
    av_reward = test(num_episodes=num_test_ep, agent=agent,
                     game_interface=hfo_interface,
                     features=features_manager, actions=actions_manager,
                     reward_funct=reward_function)
    # Save metrics structures
    trained_eps_list = [0]
    avr_epsilons_list = [agent.epsilon]
    avr_rewards_list = [av_reward]
    qlearning_variation_list = [0]
    
    # Train - test iterations:
    for i in range(num_repetitions):
        print(">>>> {}/{} <<<<".format(i, num_repetitions))
        try:
            prev_q_table = agent.q_table.copy()
            # Train:
            train(num_train_episodes=num_train_ep,
                  num_total_train_ep=num_train_ep * num_repetitions,
                  game_interface=hfo_interface, features=features_manager,
                  agent=agent, actions=actions_manager,
                  reward_funct=reward_function)
            # Update train metrics
            q_table_after_train = agent.q_table.copy()
            # Test:
            av_reward = test(num_episodes=num_test_ep, agent=agent,
                             game_interface=hfo_interface,
                             features=features_manager, actions=actions_manager,
                             reward_funct=reward_function)
        except NoActionPlayedError:
            print("\n!!! Agent was unbale to play an action !!!")
            av_reward = 0
        # except ServerDownError as e:
        #     print("\n!!! Server is Down !!!")
        #     print("iteration={}; trained_eps={}")
        #     print(str(e))
        #     break
        # check if agent trained correctly
        check_if_q_table_stayed_the_same(q_table_after_train, agent.q_table)
        sum_trained_eps = trained_eps_list[-1] + num_train_ep
        if agent.trained_eps != sum_trained_eps:
            raise Exception("Trained episodes and expected number do "
                            "not match")
        # Calc metrics:
        q_var = round(q_table_variation(prev_q_table, q_table_after_train), 4)
        print("<<TRAIN>> Q variation ", q_var)
        # Save metrics:
        trained_eps_list.append(sum_trained_eps)
        avr_epsilons_list.append(agent.epsilon)
        avr_rewards_list.append(av_reward)
        qlearning_variation_list.append(q_var)
    print("\n\n!!!!!!!!! AGENT FINISHED !!!!!!!!!!!!\n\n")
    # Save and export metrics:
    save_model(q_table=agent.q_table, file_name="agent_model",
               directory=save_dir)
    export_metrics(trained_eps=trained_eps_list, rewards=avr_rewards_list,
                   epsilons=avr_epsilons_list,
                   q_variation=qlearning_variation_list, save_dir=save_dir)
    print("\n\n!!!!!!!!! AGENT EXIT !!!!!!!!!!!!\n\n")
