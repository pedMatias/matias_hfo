import json
import os
import statistics
import argparse

import numpy as np

from matias_hfo.utils.metrics import *


# DATA_DIR = "/home/matias/Desktop/HFO/matias_hfo/data"


def get_data_from_trains(dir_path) -> (list, list, list, int):
    dirs = os.listdir(dir_path)
    epsilons_list = []
    q_table_variation_list = []
    reward_list = []
    for dir in dirs:
        if os.path.isdir(os.path.join(dir_path, dir)):
            metrics_file = os.path.join(dir_path, dir, "metrics.json")
            with open(metrics_file, "r+") as f:
                metrics = json.loads(f.read())
                epsilons = metrics["epsilons"]
                q_table_variation = [x * 100 for x in metrics[
                    "q_table_variation"]]
                reward = metrics["reward"]
                # Fill empty data:
                if metrics["trained_eps"][-1] != 10000:
                    epsilons.append(statistics.mean(epsilons[-3:]))
                    q_table_variation.append(statistics.mean(
                        q_table_variation[-3:]))
                    reward.append(statistics.mean(reward[-3:]))
                # Save data:
                epsilons_list.append(epsilons)
                q_table_variation_list.append(q_table_variation)
                reward_list.append(reward)
    return epsilons_list, q_table_variation_list, reward_list


def get_variance_of_lists(matrix: list) -> np.ndarray:
    array = np.array(matrix)
    variance = np.var(array, axis=0)
    return variance


def get_mean_of_lists(matrix: list) -> np.ndarray:
    array = np.array(matrix)
    mean = np.mean(array, axis=0)
    return mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    args = parser.parse_args()
    dir = args.dir
    
    if dir is None:
        dir = "/home/matias/Desktop/HFO/matias_hfo/data/" \
              "q_agent_train_1ep_oldEps_2020-05-10_20:30:00"
    
    # Read metrics:
    with open(dir + "/metrics.json", "r+") as f:
        metrics = json.loads(f.read())
    
    # fields:
    trained_eps = metrics["trained_eps"]
    epsilons = metrics["epsilons"]
    q_table_variation = (np.array(metrics["q_table_variation"]) * 100).tolist()
    avr_win_rate = (np.array(metrics["avr_win_rate"]) * 100).tolist()
    learning_rate = metrics["learning_rate"]
    
    num_iterations = len(trained_eps)
    
    # Create image Avr wining rate VS Learning Rate:
    chart_name = "avr_win_rate_VS_learning_rate.png"
    chart = TwoLineChart(x_legend="episodes",
                         title="Wining rate - Learning Rate")
    chart.add_first_line_chart(x=trained_eps, y=avr_win_rate,
                               name="wining rate", y_legend="wining rate (%)")
    chart.add_second_line_chart(x=trained_eps, y=learning_rate,
                                name="learning rate", y_legend="learning rate")
    chart.export_as_png(os.path.join(dir, chart_name))
    
    # Create image Avr wining rate VS Epsilon:
    chart_name = "avr_win_rate_VS_epsilon.png"
    chart = TwoLineChart(x_legend="episodes",
                         title="Wining rate (%) - Epsilon")
    chart.add_first_line_chart(x=trained_eps, y=avr_win_rate,
                               name="wining rate", y_legend="wining rate (%)")
    chart.add_second_line_chart(x=trained_eps, y=epsilons,
                                name="epsilon", y_legend="epsilon")
    chart.export_as_png(os.path.join(dir, chart_name))

    # Create image Avr wining rate VS Q-Table variation:
    chart_name = "avr_win_rate_VS_q_table_variation.png"
    chart = TwoLineChart(x_legend="episodes",
                         title="Wining rate (%) - Q-Table Variation (%)")
    chart.add_first_line_chart(x=trained_eps, y=avr_win_rate,
                               name="wining rate", y_legend="wining rate (%)")
    chart.add_second_line_chart(x=trained_eps, y=q_table_variation,
                                name="q-table variation",
                                y_legend="q-table variation (%)")
    chart.export_as_png(os.path.join(dir, chart_name))

