#!/usr/bin/hfo_env python3
# encoding utf-8
import json
import os
import argparse

from agents.plastic_v1.aux import mkdir
from agents.plastic_v1.dqn_player import Player
from agents.utils import ServerDownError


def export_metrics(trained_eps: list, avr_win_rate: list, epsilons: list,
                   model_description: str,
                   learning_rate: float, discount_factor: float, save_dir: str):
    """ Saves metrics in Json file"""
    data = {"model_description": model_description,
            "learning_rate": learning_rate, "discount_factor": discount_factor,
            "trained_eps": trained_eps,
            "epsilons": epsilons, "avr_win_rate": avr_win_rate}
    file_path = os.path.join(save_dir, "metrics.json")
    with open(file_path, 'w+') as fp:
        json.dump(data, fp)
    
        
def train_agent_online(player: Player, num_train_ep: int, num_test_ep: int,
                       starts_with_ball: bool, starts_fixed_position: bool,
                       num_repetitions: int, save_dir: str):
    # Test one first time without previous train:
    av_reward = player.test(num_episodes=num_test_ep,
                            start_with_ball=starts_with_ball,
                            starts_fixed_position=starts_fixed_position)
    # Save metrics structures
    trained_eps_list = [0]
    avr_epsilons_list = [player.agent.epsilon]
    avr_win_rate = [av_reward]
    
    # Train - test iterations:
    fail = False  # bool flag
    for i in range(num_repetitions):
        print(">>>> {}/{} <<<<".format(i, num_repetitions))
        try:
            # Train:
            av_epsilon = player.train(
                num_train_episodes=num_train_ep,
                num_total_train_ep=num_train_ep * num_repetitions,
                start_with_ball=starts_with_ball,
                starts_fixed_position=starts_fixed_position)
            # Test:
            av_reward = player.test(num_episodes=num_test_ep,
                                    start_with_ball=starts_with_ball)
        except ServerDownError as e:
            print("\n!!! Server is Down !!!")
            failed = True
            av_reward = 0
            av_epsilon = player.agent.epsilon
            player.game_interface.quit_game()
        
        sum_trained_eps = trained_eps_list[-1] + num_train_ep
        # Calc metrics:
        trained_eps_list.append(sum_trained_eps)
        avr_epsilons_list.append(av_epsilon)
        avr_win_rate.append(av_reward)
        if i % 4 == 0 or fail is True:
            export_metrics(trained_eps=trained_eps_list,
                           avr_win_rate=avr_win_rate,
                           epsilons=avr_epsilons_list,
                           model_description=player.agent.model_description,
                           learning_rate=player.agent.learning_rate,
                           discount_factor=player.agent.discount_factor,
                           save_dir=save_dir)
        if i % 5 == 0 or fail is True:
            player.agent.save_model(file_name=save_dir + "/agent_model")
    # Save and export metrics:
    export_metrics(trained_eps=trained_eps_list, avr_win_rate=avr_win_rate,
                   epsilons=avr_epsilons_list,
                   model_description=player.agent.model_description,
                   learning_rate=player.agent.learning_rate,
                   discount_factor=player.agent.discount_factor,
                   save_dir=save_dir)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--num_train_ep', type=int, default=1000)
    parser.add_argument('--num_test_ep', type=int, default=0)
    parser.add_argument('--num_repetitions', type=int, default=0)
    parser.add_argument('--retrain', type=bool, default=False)
    parser.add_argument('--starts_with_ball', type=bool, default=True)
    parser.add_argument('--starts_fixed_position', type=bool, default=True)
    parser.add_argument('--load_file', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--port', type=int, default=6000)
    
    # Parse Arguments:
    args = parser.parse_args()
    num_team = args.num_teammates
    num_op = args.num_opponents
    num_train_ep = args.num_train_ep
    num_test_ep = args.num_test_ep
    num_repetitions = args.num_repetitions
    num_episodes = (num_train_ep + num_test_ep) * num_repetitions
    starts_with_ball = args.starts_with_ball
    starts_fixed_position = args.starts_fixed_position
    port = args.port

    # Start Player:
    player = Player(num_teammates=num_team, num_opponents=num_op, port=port)
    
    # IF retrain mode, load previous model
    if args.retrain and args.load_file:
        print("[RETRAIN] Loading Model! ")
        player.agent.load_model(args.load_file)
        dir_base_name = "retrain" if args.retrain else "new"
    else:
        print("[TRAIN] New Model!")
        dir_base_name = "new"
   
    # Directory
    save_dir = args.save_dir or mkdir(name="dqn", num_episodes=num_episodes,
                                      num_op=num_op, extra_note=dir_base_name)
    
    # Save original model:
    if args.retrain:
        print("[RETRAIN] Save original model")
        player.agent.save_model(file_name=save_dir + "/original_agent_model")
        
    print("\n[{} - PLAYER] num_opponents={}; num_teammates={}; "
          "start_with_ball={}".format(
            "RETRAIN" if args.retrain else "TRAIN",
            num_op, num_op, starts_with_ball))
    
    train_agent_online(player=player, num_train_ep=num_train_ep,
                       num_test_ep=num_test_ep,
                       starts_with_ball=starts_with_ball,
                       starts_fixed_position=starts_fixed_position,
                       num_repetitions=num_repetitions, save_dir=save_dir)
    
    print("\n\n!!!!!!!!! AGENT FINISHED !!!!!!!!!!!!\n\n")
    player.agent.save_model(file_name=save_dir + "/agent_model")
    print("\n\n!!!!!!!!! AGENT EXIT !!!!!!!!!!!!\n\n")