#!/usr/bin/hfo_env python3
# encoding utf-8
import json
import os
import argparse

from agents.agent_module_dqn.aux import mkdir
from agents.agent_module_dqn.player import Player
from agents.utils import ServerDownError


def export_metrics(trained_eps: list, avr_win_rate: list, epsilons: list,
                   save_dir: str):
    """ Saves metrics in Json file"""
    data = {"trained_eps": trained_eps, "epsilons": epsilons,
            "avr_win_rate": avr_win_rate}
    file_path = os.path.join(save_dir, "metrics.json")
    with open(file_path, 'w+') as fp:
        json.dump(data, fp)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_train_ep', type=int, default=1000)
    parser.add_argument('--num_test_ep', type=int, default=0)
    parser.add_argument('--num_repetitions', type=int, default=0)
    parser.add_argument('--retrain', type=bool, default=False)
    parser.add_argument('--starts_with_ball', type=bool, default=True)
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
        player.agent.load_model(args.load_file)
   
    # Directory
    save_dir = args.save_dir or mkdir(
        num_episodes, num_op,
        extra_note="retrain" if args.retrain else "new")

    print("\n[{} - PLAYER] num_opponents={}; num_teammates={}; "
          "start_with_ball={}".format("RETRAIN" if args.retrain else "TRAIN",
                                      num_op, num_op, starts_with_ball))
    
    # Test one first time without previous train:
    # av_reward = player.test(num_episodes=num_test_ep,
    #                         start_with_ball=starts_with_ball)
    # Save metrics structures
    trained_eps_list = [0]
    avr_epsilons_list = [player.agent.epsilon]
    avr_win_rate = [0]  # av_reward]
    
    # Train - test iterations:
    for i in range(num_repetitions):
        print(">>>> {}/{} <<<<".format(i, num_repetitions))
        try:
            # Train:
            player.train(num_train_episodes=num_train_ep,
                         num_total_train_ep=num_train_ep * num_repetitions,
                         start_with_ball=starts_with_ball)
            # Test:
            av_reward = player.test(num_episodes=num_test_ep,
                                    start_with_ball=starts_with_ball)
        except ServerDownError as e:
            print("\n!!! Server is Down !!!")
            pass
            av_reward = 0
        sum_trained_eps = trained_eps_list[-1] + num_train_ep
        # Calc metrics:
        trained_eps_list.append(sum_trained_eps)
        avr_epsilons_list.append(player.agent.epsilon)
        avr_win_rate.append(av_reward)
    print("\n\n!!!!!!!!! AGENT FINISHED !!!!!!!!!!!!\n\n")
    # Save and export metrics:
    player.agent.save_model(file_name=save_dir + "/agent_model")
    export_metrics(trained_eps=trained_eps_list, avr_win_rate=avr_win_rate,
                   epsilons=avr_epsilons_list, save_dir=save_dir)
    print("\n\n!!!!!!!!! AGENT EXIT !!!!!!!!!!!!\n\n")