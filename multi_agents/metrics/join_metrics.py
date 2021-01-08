import json
import os
import random

import numpy as np
import scipy.stats as st

from multi_agents import config

SIZE = 25
selected_teams_file_format = "{team_name}_t{idx}_plastic_agent1_" \
                             "selected_teams.json"

TEAMS_NAMES = config.TEAMS_NAMES + ["base"]


def matrix_mean_confidence_interval(matrix: np.ndarray, confidence=0.9) \
        -> (list, list):
    n = matrix.shape[1]
    m = np.mean(matrix, axis=0)
    se = st.sem(matrix, axis=0)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m.tolist(), h.tolist()


def array_mean_confidence_interval(array: np.ndarray, confidence=0.9) \
        -> (list, list):
    n = len(array)
    m = np.mean(array)
    se = st.sem(array)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m.tolist(), h.tolist()


def load_game_results(result_file: str, team_dir: str):
    result_file = os.path.join(team_dir, result_file)
    if os.path.isfile(result_file):
        with open(result_file) as json_file:
            return json.load(json_file)
    else:
        print("File {} not found!".format(result_file))
        return None


def load_selected_teams_results(result_file: str, team_dir: str,
                                correct_team: str):
    result_file = os.path.join(team_dir, result_file)
    if os.path.isfile(result_file):
        with open(result_file) as json_file:
            selected_teams = json.load(json_file)
        if len(selected_teams) < (SIZE - SIZE/5):
            print(f"[Too much missing data] File {result_file}")
            return None, None
        
        beliefs_correct_team = []
        prob_model_max = []
        for selected_teams_it in selected_teams:
            belief = selected_teams_it[correct_team]
            if belief >= max(selected_teams_it.values()):
                prob_model_max.append(1)
            else:
                prob_model_max.append(0)
            beliefs_correct_team.append(belief)
        
        # Fill missing data:
        if len(beliefs_correct_team) >= SIZE:
            return beliefs_correct_team[:SIZE], prob_model_max[:SIZE]
        else:
            if len(beliefs_correct_team) <= (SIZE - SIZE/5):
                return None, None
            num_left_values = SIZE - len(beliefs_correct_team)
            last_values = beliefs_correct_team[-num_left_values * 3:]
            rand_values = random.choices(last_values, k=num_left_values)
            beliefs_correct_team += rand_values

            last_values = prob_model_max[-num_left_values * 3:]
            rand_values = random.choices(last_values, k=num_left_values)
            prob_model_max += rand_values
            assert len(prob_model_max) == SIZE
            return beliefs_correct_team, prob_model_max
    else:
        print("File {} not found!".format(result_file))
        return None, None


def fill_missing_data(game_result_list: list):
    num_left_values = SIZE - len(game_result_list)
    last_values = game_result_list[-num_left_values * 3:]
    rand_values = random.choices(last_values, k=num_left_values)
    game_result_list += rand_values
    assert len(game_result_list) == SIZE
    return game_result_list

  
def collect_plastic_agent_temporal_data():
    BASE_DIR = "/home/matias/Desktop/HFO/matias_hfo/models/4vs5/metrics"
    results_path = BASE_DIR + "/plastic/{team_type}/{agent_type}"
    metrics_path = BASE_DIR + "/tese/{team_type}/{agent_type}"
    game_results_file_format = "{team_name}_t{idx}_plastic1_game_results.json"
    
    for teammate_type in ["w_stochastic_plastic_agent",
                          "w_adversarial_plastic_agent", "w_npc"]:
        for agent_type in ["adversarial", "stochastic"]:
            if "w_stochastic_plastic_agent" == teammate_type and \
                    agent_type == "adversarial":
                continue
            if "w_adversarial_plastic_agent" == teammate_type and \
                    agent_type == "stochastic":
                continue
                
            # Set up data auxiliar structures:
            all_teams_games_results = []
            teams_results = {}
            for team in TEAMS_NAMES:
                teams_results[team] = []
                
            directory_2 = results_path.format(team_type=teammate_type,
                                              agent_type=agent_type)
            assert os.path.isdir(directory_2)
            sub_dirs = os.listdir(directory_2)
            for sub_dir in sub_dirs:
                # For each team:
                for team in TEAMS_NAMES:
                    idx = 0
                    not_found_ep = 0
                    games_results = []
                    # Search for all the tests:
                    while True:
                        result_file = game_results_file_format.format(
                            team_name=team, idx=idx)
                        game_result = load_game_results(
                            result_file=result_file,
                            team_dir=os.path.join(directory_2, sub_dir, team)
                        )
                        # Found test files:
                        if game_result:
                            if len(game_result) < SIZE:
                                game_result = fill_missing_data(game_result)
                            else:
                                game_result = game_result[:SIZE]
                            assert len(game_result) == SIZE
                            
                            not_found_ep = 0
                            games_results.append(game_result)
                            all_teams_games_results.append(game_result)
                        # Not found files:
                        else:
                            not_found_ep += 1
            
                        idx += 1
                        # Not found more than 5 episodes, assume the test ended
                        if not_found_ep >= 5:
                            break
                    teams_results[team] += games_results
            
            print(f"[{teammate_type}] Type {agent_type} data: "
                  f"{len(all_teams_games_results)}")
            save_dir = metrics_path.format(team_type=teammate_type,
                                           agent_type=agent_type)
            # Process team data:
            for team in TEAMS_NAMES:
                if not teams_results[team]:
                    continue
                team_game_results = np.array(teams_results[team])
                mean_vs, conf_int = matrix_mean_confidence_interval(
                    team_game_results)
                game_results_dict = dict(mean_values=mean_vs,
                                         confidence_int=conf_int)
            
                # Save metrics:
                new_metrics_file = team + "_metrics.json"
                new_metrics_file = os.path.join(save_dir, new_metrics_file)
                with open(new_metrics_file, "w+") as new_file:
                    json.dump(game_results_dict, new_file)
    
            # Process All the teams data:
            all_teams_games_results = np.array(all_teams_games_results)
            mean_vs, conf_int = matrix_mean_confidence_interval(
                all_teams_games_results)
            all_teams_game_results_dict = dict(mean_values=mean_vs,
                                               confidence_int=conf_int)
        
            # Save metrics:
            new_metrics_file = "all_metrics.json"
            new_metrics_file = os.path.join(save_dir, new_metrics_file)
            with open(new_metrics_file, "w+") as new_file:
                json.dump(all_teams_game_results_dict, new_file)


def collect_plastic_agent_sum_data():
    BASE_DIR = "/home/matias/Desktop/HFO/matias_hfo/models/4vs5/metrics"
    results_path = BASE_DIR + "/plastic/{team_type}/{agent_type}"
    metrics_path = BASE_DIR + "/tese/{team_type}/{agent_type}"
    game_results_file_format = "{team_name}_t{idx}_plastic1_game_results.json"

    for teammate_type in ["w_stochastic_plastic_agent",
                          "w_adversarial_plastic_agent", "w_npc"]:
        for agent_type in ["adversarial", "stochastic"]:
            if "w_stochastic_plastic_agent" == teammate_type and \
                    agent_type == "adversarial":
                continue
            if "w_adversarial_plastic_agent" == teammate_type and \
                    agent_type == "stochastic":
                continue
            # Set up data auxiliar structures:
            all_teams_games_results = []
            teams_results = {}
            for team in TEAMS_NAMES:
                teams_results[team] = []
            
            directory_2 = results_path.format(team_type=teammate_type,
                                              agent_type=agent_type)
            print("Dir2: ", directory_2)
            assert os.path.isdir(directory_2)
            sub_dirs = os.listdir(directory_2)
            for sub_dir in sub_dirs:
                # For each team:
                for team in TEAMS_NAMES:
                    idx = 0
                    not_found_ep = 0
                    games_results = []
                    # Search for all the tests:
                    while True:
                        result_file = game_results_file_format.format(
                            team_name=team, idx=idx)
                        game_result = load_game_results(
                            result_file=result_file,
                            team_dir=os.path.join(directory_2, sub_dir, team)
                        )
                        # Found test files:
                        if game_result:
                            if len(game_result) < SIZE:
                                game_result = fill_missing_data(game_result)
                            else:
                                game_result = game_result[:SIZE + 1]
                            not_found_ep = 0
                            games_results.append(sum(game_result))
                            all_teams_games_results.append(sum(game_result))
                        # Not found files:
                        else:
                            not_found_ep += 1
                        
                        idx += 1
                        # Not found more than 5 episodes, assume the test ended
                        if not_found_ep >= 5:
                            break
                    teams_results[team] += games_results
            
            print(f"[{teammate_type}] Type {agent_type} data: "
                  f"{len(all_teams_games_results)}")
            save_dir = metrics_path.format(team_type=teammate_type,
                                           agent_type=agent_type)
            # Process team data:
            for team in TEAMS_NAMES:
                if not teams_results[team]:
                    continue
                team_game_results = np.array(teams_results[team])
                mean_vs, conf_int = array_mean_confidence_interval(
                    team_game_results)
                game_results_dict = dict(mean_value=mean_vs,
                                         confidence_int=conf_int,
                                         values=team_game_results.tolist())
                
                # Save metrics:
                new_metrics_file = "sum_" + team + "_metrics.json"
                new_metrics_file = os.path.join(save_dir, new_metrics_file)
                with open(new_metrics_file, "w+") as new_file:
                    json.dump(game_results_dict, new_file)
            
            # Process All the teams data:
            all_teams_games_results = np.array(all_teams_games_results)
            mean_vs, conf_int = array_mean_confidence_interval(
                all_teams_games_results)
            all_teams_game_results_dict = dict(
                mean_value=mean_vs,
                confidence_int=conf_int,
                values=all_teams_games_results.tolist())
            
            # Save metrics:
            new_metrics_file = "sum_all_metrics.json"
            new_metrics_file = os.path.join(save_dir, new_metrics_file)
            with open(new_metrics_file, "w+") as new_file:
                json.dump(all_teams_game_results_dict, new_file)


def collect_plastic_selected_teams_data():
    BASE_DIR = "/home/matias/Desktop/HFO/matias_hfo/models/4vs5/metrics"
    results_path = BASE_DIR + "/plastic/{team_type}/{agent_type}"
    metrics_path = BASE_DIR + "/tese/{team_type}/{agent_type}"
    game_results_file_format = "{team_name}_t{idx}_plastic1_" \
                               "selected_teams.json"
    
    teammate_type = "w_npc"
    for agent_type in ["adversarial", "stochastic"]:
        directory_2 = results_path.format(team_type=teammate_type,
                                          agent_type=agent_type)
        correct_team_beliefs = []
        prob_model_maxs = []
        
        assert os.path.isdir(directory_2)
        sub_dirs = os.listdir(directory_2)
        for sub_dir in sub_dirs:
            # For each team:
            for team in TEAMS_NAMES:
                idx = 0
                not_found_ep = 0
                # Search for all the tests:
                while True:
                    result_file = game_results_file_format.format(
                        team_name=team, idx=idx)
                    game_result, prob_model_max = load_selected_teams_results(
                        result_file=result_file,
                        team_dir=os.path.join(directory_2, sub_dir, team),
                        correct_team=team
                    )
                    # Found test files:
                    if game_result:
                        not_found_ep = 0
                        correct_team_beliefs.append(game_result)
                        prob_model_maxs.append(prob_model_max)
                    # Not found files:
                    else:
                        not_found_ep += 1
                    
                    idx += 1
                    # Not found more than 5 episodes, assume the test ended
                    if not_found_ep >= 5:
                        break
        
        print(f"[{teammate_type}] Type {agent_type} data: "
              f"{len(correct_team_beliefs)}")
        save_dir = metrics_path.format(team_type=teammate_type,
                                       agent_type=agent_type)
        # Process team data:
        team_game_results = np.array(correct_team_beliefs)
        beliefs_mean_vs, beliefs_conf_int = matrix_mean_confidence_interval(
            team_game_results)
        prob_model_maxs = np.array(prob_model_maxs)
        max_mean_vs, max_conf_int = matrix_mean_confidence_interval(
            prob_model_maxs)
        game_results_dict = dict(
            beliefs_mean_value=beliefs_mean_vs,
            beliefs_confidence_int=beliefs_conf_int,
            prob_correct_model_mean_value=max_mean_vs,
            prob_correct_model_mean_int=max_conf_int
        )
        
        # Save metrics:
        new_metrics_file = "beliefs_metrics.json"
        new_metrics_file = os.path.join(save_dir, new_metrics_file)
        with open(new_metrics_file, "w+") as new_file:
            json.dump(game_results_dict, new_file)

"""
def collect_random_agent_temporal_data():
    BASE_DIR = "/home/matias/Desktop/HFO/matias_hfo/models/4vs5/metrics"
    results_path = BASE_DIR + "/random/{team_type}"
    metrics_path = BASE_DIR + "/tese/random/{team_type}"
    game_results_file_format = "{team_name}_t{idx}_random1_game_results.json"
    
    for teammate_type in ["w_plastic_agent", "w_npc"]:
        # Set up data auxiliar structures:
        all_teams_games_results = []
        teams_results = {}
        for team in TEAMS_NAMES:
            teams_results[team] = []
        
        directory_2 = results_path.format(team_type=teammate_type)
        assert os.path.isdir(directory_2)
        
        sub_dirs = os.listdir(directory_2)
        for sub_dir in sub_dirs:
            # For each team:
            for team in TEAMS_NAMES:
                # If not found team:
                if not os.path.isdir(os.path.join(directory_2, sub_dir, team)):
                    print(f"[{teammate_type}] {team} not found")
                    continue
                    
                idx = 0
                not_found_ep = 0
                games_results = []
                # Search for all the tests:
                while True:
                    result_file = game_results_file_format.format(
                        team_name=team, idx=idx)
                    game_result = load_game_results(
                        result_file=result_file,
                        team_dir=os.path.join(directory_2, sub_dir, team)
                    )
                    # Found test files:
                    if game_result:
                        if len(game_result) < SIZE:
                            game_result = fill_missing_data(game_result)
                        not_found_ep = 0
                        games_results.append(game_result)
                        all_teams_games_results.append(game_result)
                    # Not found files:
                    else:
                        not_found_ep += 1
                    
                    idx += 1
                    # Not found more than 5 episodes, assume the test ended
                    if not_found_ep >= 5:
                        break
                teams_results[team] += games_results
        
        print(f"[{teammate_type}] data: {len(all_teams_games_results)}")
        save_dir = metrics_path.format(team_type=teammate_type)
        # Process team data:
        for team in TEAMS_NAMES:
            if not teams_results[team]:
                continue
            team_game_results = np.array(teams_results[team])
            mean_vs, conf_int = mean_confidence_interval(team_game_results)
            game_results_dict = dict(mean_values=mean_vs,
                                     confidence_int=conf_int)
            
            # Save metrics:
            new_metrics_file = team + "_metrics.json"
            new_metrics_file = os.path.join(save_dir, new_metrics_file)
            with open(new_metrics_file, "w+") as new_file:
                json.dump(game_results_dict, new_file)
        
        # Process All the teams data:
        all_teams_games_results = np.array(all_teams_games_results)
        mean_vs, conf_int = mean_confidence_interval(
            all_teams_games_results)
        all_teams_game_results_dict = dict(mean_values=mean_vs,
                                           confidence_int=conf_int)
        
        # Save metrics:
        new_metrics_file = "all_metrics.json"
        new_metrics_file = os.path.join(save_dir, new_metrics_file)
        with open(new_metrics_file, "w+") as new_file:
            json.dump(all_teams_game_results_dict, new_file)


def collect_random_agent_sum_data():
    BASE_DIR = "/home/matias/Desktop/HFO/matias_hfo/models/4vs5/metrics"
    results_path = BASE_DIR + "/random/{team_type}"
    metrics_path = BASE_DIR + "/tese/random/{team_type}"
    game_results_file_format = "{team_name}_t{idx}_random1_game_results.json"
    
    for teammate_type in ["w_plastic_agent", "w_npc"]:
        # Set up data auxiliar structures:
        all_teams_games_results = []
        teams_results = {}
        for team in TEAMS_NAMES:
            teams_results[team] = []
        
        directory_2 = results_path.format(team_type=teammate_type)
        assert os.path.isdir(directory_2)
        
        sub_dirs = os.listdir(directory_2)
        for sub_dir in sub_dirs:
            # For each team:
            for team in TEAMS_NAMES:
                # If not found team:
                if not os.path.isdir(os.path.join(directory_2, sub_dir, team)):
                    print(f"[{teammate_type}] {team} not found")
                    continue
                
                idx = 0
                not_found_ep = 0
                games_results = []
                # Search for all the tests:
                while True:
                    result_file = game_results_file_format.format(
                        team_name=team, idx=idx)
                    game_result = load_game_results(
                        result_file=result_file,
                        team_dir=os.path.join(directory_2, sub_dir, team)
                    )
                    # Found test files:
                    if game_result:
                        if len(game_result) < SIZE:
                            game_result = fill_missing_data(game_result)
                        else:
                            game_result = game_result[:SIZE+1]
                        not_found_ep = 0
                        games_results.append(sum(game_result))
                        all_teams_games_results.append(sum(game_result))
                    # Not found files:
                    else:
                        not_found_ep += 1
                    
                    idx += 1
                    # Not found more than 5 episodes, assume the test ended
                    if not_found_ep >= 5:
                        break
                teams_results[team] += games_results
        
        print(f"[{teammate_type}] num_games: {len(all_teams_games_results)}")
        save_dir = metrics_path.format(team_type=teammate_type)
        # Process team data:
        for team in TEAMS_NAMES:
            if not teams_results[team]:
                continue
            team_game_results = np.array(teams_results[team])
            mean_vs, conf_int = array_mean_confidence_interval(
                team_game_results)
            game_results_dict = dict(mean_values=mean_vs,
                                     confidence_int=conf_int)
            
            # Save metrics:
            new_metrics_file = "sum_" + team + "_metrics.json"
            new_metrics_file = os.path.join(save_dir, new_metrics_file)
            with open(new_metrics_file, "w+") as new_file:
                json.dump(game_results_dict, new_file)
        
        # Process All the teams data:
        all_teams_games_results = np.array(all_teams_games_results)
        mean_vs, conf_int = array_mean_confidence_interval(
            all_teams_games_results)
        all_teams_game_results_dict = dict(mean_values=mean_vs,
                                           confidence_int=conf_int)
        
        # Save metrics:
        new_metrics_file = "sum_all_metrics.json"
        new_metrics_file = os.path.join(save_dir, new_metrics_file)
        with open(new_metrics_file, "w+") as new_file:
            json.dump(all_teams_game_results_dict, new_file)
"""

if __name__ == '__main__':
    collect_plastic_selected_teams_data()
