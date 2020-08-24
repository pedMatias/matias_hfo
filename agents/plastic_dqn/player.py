#!/usr/bin/env python3
# encoding utf-8
import argparse
import json
import os
import pickle
import random
from collections import deque

from agents.offline_plastic_v2.base.hfo_attacking_player import \
    HFOAttackingPlayer
from agents.offline_plastic_v2.deep_agent import DQNAgent
from agents.offline_plastic_v2.actions.complex import Actions
from agents.offline_plastic_v2.features.plastic_features import PlasticFeatures
from agents.offline_plastic_v2.aux import check_same_model


STARTING_POSITIONS = {"TOP LEFT": (-0.5, -0.7), "TOP RIGHT": (0.4, -0.7),
                      "MID LEFT": (-0.5, 0.0), "MID RIGHT": (0.4, 0.0),
                      "BOTTOM LEFT": (-0.5, 0.7), "BOTTOM RIGHT": (0.4, 0.7)}

DISCOUNT = 0.99
LEARNING_RATE = 0.00025
REPLAY_MEMORY_SIZE = 1_000_000  # How many last steps to keep for model training
NUM_EPOCHS = 10
NUM_TRAIN_REP = 50


class Player:
    def __init__(self, num_opponents: int, num_teammates: int,
                 learn_team: bool = True, verbose: bool = True,
                 port: int = 6000):
        self.num_teammates = num_teammates
        self.num_opponents = num_opponents
        self.learn_team = learn_team
        
        # Game Interface:
        self.game_interface = HFOAttackingPlayer(num_opponents=num_opponents,
                                                 num_teammates=num_teammates,
                                                 port=port)
        # Features Interface:
        self.features = PlasticFeatures(num_op=num_opponents,
                                        num_team=num_teammates)
        # Actions Interface:
        self.actions = Actions(num_team=num_teammates, features=self.features,
                               game_interface=self.game_interface)
        # Agent instance:
        self.agent = DQNAgent(num_features=self.features.num_features,
                              num_actions=self.actions.get_num_actions(),
                              learning_rate=LEARNING_RATE,
                              discount_factor=DISCOUNT)
        
        
def train_mode_1(train_data: list, player: Player, save_model_file: str):
    # Use 10% of the dataset
    if len(train_data) > 100000:
        train_batch_size = 10000
    else:
        train_batch_size = int(len(train_data) * 0.1)
    
    # Minimum train iterations:
    min_train_iter = 5
    
    saved_iteration = 0
    all_losses = []
    usefull_losses = []
    num_rep = 20
    counter_num_stable_trains = 0
    for i in range(num_rep):
        # if counter_num_stable_trains >= 2:
        #     break
        
        # Fit train_data:
        batch = random.sample(train_data, train_batch_size)
        loss = player.agent.fit_batch(batch, verbose=0, epochs=150)
        
        # Loss:
        avr_loss = sum(loss) / len(loss)
        print(f"[{i}/{num_rep}] Average loss {avr_loss}")
        
        if i > min_train_iter:
            if len(usefull_losses) == 0 or avr_loss < min(usefull_losses):
                player.agent.save_model(file_name=save_model_file)
                saved_iteration = i
            
            if len(usefull_losses) > 0:
                changed_percentage = (avr_loss * 100) / usefull_losses[-1]
                changed_percentage = 100 - changed_percentage
                # Check if loss changed less than 3%
                if changed_percentage < 3:
                    counter_num_stable_trains += 1
                    print(
                        f"[{i}/{num_rep}]Loss variation {changed_percentage}")
            usefull_losses.append(avr_loss)
        all_losses.append(avr_loss)
    
    else:
        player.agent.save_model(file_name=save_model_file)
        saved_iteration = num_rep
    return all_losses, saved_iteration


def train_mode_2(train_data: list, player: Player, save_model_file: str):
    saved_iteration = 0
    losses = []
    num_rep = 100
    min_trains = 50
    num_min_stable_trains = 10
    c_num_stable_trains = 0
    
    num_epochs = 5
    
    for i in range(num_rep):
        # Train:
        loss = player.agent.fit_batch(train_data, verbose=0, epochs=num_epochs)
        # Avr Loss:
        avr_loss = sum(loss) / len(loss)
        print(f"[{i}/{num_rep}] Loss {avr_loss}")
        if i < min_trains:
            continue
        else:
            if c_num_stable_trains >= num_min_stable_trains:
                break
            # Min loss:
            min_loss = min(losses) if losses else avr_loss
            # Low loss:
            if avr_loss <= min_loss:
                player.agent.save_model(file_name=save_model_file)
                saved_iteration = i
            
            if len(losses) > 0:
                changed_percentage = (avr_loss * 100) / losses[-1]
                changed_percentage = 100 - changed_percentage
                # Check if loss changed less than 3%
                if changed_percentage < 1:
                    c_num_stable_trains += 1
                    print(f"[{i}/{num_rep}]Loss variation "
                          f"{changed_percentage}")
            losses.append(avr_loss)
    else:
        player.agent.save_model(file_name=save_model_file)
        saved_iteration = num_rep
    return losses, saved_iteration


def train_mode_3(train_data: deque, player: Player, save_model_file: str,
                 save_all: bool = False, num_rep: int = NUM_TRAIN_REP):
    saved_iterations = []
    losses = []
    c_num_stable_trains = 0
    
    num_min_stable_trains = 10
    
    for i in range(num_rep):
        # Early stop:
        if c_num_stable_trains >= num_min_stable_trains:
            break
            
        # Train:
        train_data = list(train_data)
        loss = player.agent.fit_batch(train_data, verbose=0, epochs=NUM_EPOCHS)
        
        # Avr Loss:
        avr_loss = sum(loss) / len(loss)
        print(f"[{i}/{num_rep}] Loss {avr_loss}")
        
        if len(losses) > 0:
            # Save model:
            if save_all:
                # Save over last saved:
                saved_iterations.append(i)
                new_model_file = save_model_file + "." + str(
                    len(saved_iterations))
                player.agent.save_model(file_name=new_model_file)
                print(f"[{i}/{num_rep}] SAVE MODEL {new_model_file}")
                
            elif avr_loss < min(losses) or avr_loss < min(losses[i-5:i]):
                # Save over last saved:
                if i-1 in saved_iterations:
                    saved_iterations[-1] = i
                    new_model_file = save_model_file + "." + str(
                        len(saved_iterations))
                else:
                    saved_iterations.append(i)
                    new_model_file = save_model_file + "." + str(
                        len(saved_iterations))
                player.agent.save_model(file_name=new_model_file)
                print(f"[{i}/{num_rep}] SAVE MODEL {new_model_file}")

            # Check if loss changed less than 1%
            changed_percentage = (avr_loss * 100) / losses[-1]
            changed_percentage = 100 - changed_percentage
            if 0 < changed_percentage < 3:
                c_num_stable_trains += 1
                print(f"[{i}/{num_rep}]Loss variation "
                      f"{changed_percentage}")
            else:
                c_num_stable_trains = 0
        losses.append(avr_loss)
    else:
        saved_iterations.append(num_rep)
        new_model_file = save_model_file + "." + str(len(saved_iterations))
        player.agent.save_model(file_name=new_model_file)
    return losses, saved_iterations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_opponents', type=int, default=0)
    parser.add_argument('--num_teammates', type=int, default=0)
    parser.add_argument('--replay_buffer_size', type=int,
                        default=REPLAY_MEMORY_SIZE)
    parser.add_argument('--num_train_rep', type=int, default=NUM_TRAIN_REP)
    parser.add_argument('--new_model', type=str, default="false")
    parser.add_argument('--save_all', type=str, default="false")
    parser.add_argument('--dir', type=str)
    parser.add_argument('--stage', type=float)
    
    # Parse Arguments:
    args = parser.parse_args()
    print(f"\n[TRAIN: set-up] {args}\n")
    num_team = args.num_teammates
    num_op = args.num_opponents
    replay_buffer_size = args.replay_buffer_size
    num_train_rep = args.num_train_rep
    new_model = args.new_model
    save_all = True if args.save_all == "true" else False
    directory = args.dir
    stage = args.stage
    
    # Start Player:
    player = Player(num_teammates=num_team, num_opponents=num_op)
    
    # Load Train data:
    train_data = deque(maxlen=replay_buffer_size)
    var1 = 1.0
    while var1 <= stage:
        data_file = os.path.join(directory, f"learn_buffer_{str(var1)}")
        print(data_file)
        if os.path.isfile(data_file):
            with open(data_file, "rb") as fp:
                data = pickle.load(fp)
                print(f"Add stage {var1} data. SIZE={len(data)}")
                train_data += data
            var1 += 0.1  # Check next sub-stage
            var1 = round(var1, 1)
        else:
            var1 += 1  # Check next stage
            var1 = float(int(var1))

    print(f"\n[TRAIN OFFLINE: Stage {stage}] DATA LEN={len(train_data)};\n")
    
    # Load Model if stage higher than 1:
    if stage > 1 and new_model == "false":
        # Beginning of a stage:
        if stage % 1 == 0:
            prev_sub_stage = int(stage - 1)
        else:
            prev_sub_stage = stage - 0.1
            prev_sub_stage = round(prev_sub_stage, 1)
        
        # Get model file:
        model_file = os.path.join(directory, f"agent_model_{prev_sub_stage}")
        
        if not os.path.isfile(model_file):
            print("[LOAD MODEL File] Cant find previous Model!!")
            pass
        else:
            print("[LOAD MODEL File] Load model {}".format(model_file))
            player.agent.load_model(model_file)
    else:
        print("[LOAD MODEL File] Create new Model!!")
    
    # Save model file name:
    sub_stage_model_file = os.path.join(directory, f"agent_model_{stage}")
    
    # TRAIN model:
    losses, saved_iterations = train_mode_3(
        train_data=train_data,
        player=player,
        save_model_file=sub_stage_model_file,
        save_all=save_all,
        num_rep=num_train_rep
    )

    # Train metrics data:
    train_metrics_file = os.path.join(directory, "train_metrics.json")
    if stage > 1 and os.path.isfile(train_metrics_file):
        with open(train_metrics_file, "rb") as fp:
            train_metrics = json.load(fp)
    else:
        train_metrics = dict()
    
    # Write train metrics:
    train_metrics[f"stage_{stage}"] = {
        "learning_rate": player.agent.learning_rate,
        "saved_iterations": saved_iterations,
        "train_data_size": len(train_data),
        "avr_loss": sum(losses)/len(losses)}
    
    with open(train_metrics_file, 'w+') as fp:
        json.dump(train_metrics, fp)
    
    print("\n!!!!!!!!! Train End !!!!!!!!!!!!\n\n")
