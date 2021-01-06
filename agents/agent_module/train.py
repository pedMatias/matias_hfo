#!/usr/bin/hfo_env python3
# encoding utf-8
import json
import os
import argparse

from utils.aux_functions import q_table_variation
from agents.agent_module.aux import mkdir, save_model, ServerDownError
from agents.agent_module.player import Player


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
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_train_ep', type=int, default=1000)
    parser.add_argument('--num_test_ep', type=int, default=0)
    parser.add_argument('--num_repetitions', type=int, default=0)
    parser.add_argument('--retrain', type=bool, default=False)
    parser.add_argument('--starts_with_ball', type=bool, default=False)
    parser.add_argument('--load_file', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    # Parse Arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_train_ep = args.num_train_ep
    num_test_ep = args.num_test_ep
    num_repetitions = args.num_repetitions
    num_episodes = (num_train_ep + num_test_ep) * num_repetitions
    starts_with_ball = args.starts_with_ball

    # Start Player:
    player = Player(num_teammates=num_team, num_opponents=num_op)
    
    # IF retrain mode, load previous model
    if args.retrain and args.load_file:
        player.agent.load_q_table(args.load_file)
   
    # Directory
    save_dir = args.save_dir or mkdir(
        num_episodes, num_op,
        extra_note="retrain" if args.retrain else "new")

    print("\n[{} - PLAYER] num_opponents={}; num_teammates={}; "
          "start_with_ball={}".format("RETRAIN" if args.retrain else "TRAIN",
                                      num_op, num_op, starts_with_ball))
    
    # Test one first time without previous train:
    av_reward = player.test(num_episodes=num_test_ep,
                            start_with_ball=starts_with_ball)
    # Save metrics structures
    trained_eps_list = [0]
    avr_epsilons_list = [player.agent.epsilon]
    avr_rewards_list = [av_reward]
    qlearning_variation_list = [0]
    
    # Train - test iterations:
    for i in range(num_repetitions):
        print(">>>> {}/{} <<<<".format(i, num_repetitions))
        try:
            prev_q_table = player.agent.q_table.copy()
            # Train:
            player.train(num_train_episodes=num_train_ep,
                         num_total_train_ep=num_train_ep * num_repetitions,
                         start_with_ball=starts_with_ball)
            # Update train metrics
            q_table_after_train = player.agent.q_table.copy()
            # Test:
            av_reward = player.test(num_episodes=num_test_ep,
                                    start_with_ball=starts_with_ball)
        except ServerDownError as e:
            print("\n!!! Server is Down !!!")
        sum_trained_eps = trained_eps_list[-1] + num_train_ep
        # Calc metrics:
        q_var = round(q_table_variation(prev_q_table, q_table_after_train), 4)
        print("<<TRAIN>> Q variation ", q_var)
        # Save metrics:
        trained_eps_list.append(sum_trained_eps)
        avr_epsilons_list.append(player.agent.epsilon)
        avr_rewards_list.append(av_reward)
        qlearning_variation_list.append(q_var)
    print("\n\n!!!!!!!!! AGENT FINISHED !!!!!!!!!!!!\n\n")
    # Save and export metrics:
    save_model(q_table=player.agent.q_table, file_name="agent_model",
               directory=save_dir)
    export_metrics(trained_eps=trained_eps_list, rewards=avr_rewards_list,
                   epsilons=avr_epsilons_list,
                   q_variation=qlearning_variation_list, save_dir=save_dir)
    print("\n\n!!!!!!!!! AGENT EXIT !!!!!!!!!!!!\n\n")