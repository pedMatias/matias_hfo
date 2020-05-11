import json
import os
import statistics

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
                q_table_variation = [x*100 for x in metrics[
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
    dir = "/home/matias/Desktop/HFO/matias_hfo/data/q_agent_stable_version_1lr_low_ep_start_5_qtable"
    epsilons_l, q_table_variation_l, reward_l = get_data_from_trains(dir)
    num_instances = len(epsilons_l)
    x = list(range(0, 10001, 500))
    # Epsilon:
    epsilons_mean_l = get_mean_of_lists(epsilons_l)
    # Learning rate:
    lr_mean_l = np.array([0.1] * 21)
    # Q_variation:
    q_table_variation_mean_l = get_mean_of_lists(q_table_variation_l)
    q_table_variation_lower_l = q_table_variation_mean_l - \
        get_variance_of_lists(q_table_variation_l)
    q_table_variation_upper_l = q_table_variation_mean_l + \
        get_variance_of_lists(q_table_variation_l)
    # Reward:
    reward_mean_l = get_mean_of_lists(reward_l)
    reward_lower_l = reward_mean_l - get_variance_of_lists(reward_l)
    reward_upper_l = reward_mean_l + get_variance_of_lists(reward_l)
    
    # Create image Qvariantion-epsilon:
    chart_name = "epsilon_q_variation.png"
    chart = TwoLineChart(x_legend="episodes", title="Average q learning "
                         "variation of {} training iterations".
                         format(num_instances))
    chart.add_first_line_chart(x=x, y=epsilons_mean_l.tolist(),
                               name="epsilon", y_legend="epsilon")
    chart.add_second_line_chart(x=x, y=q_table_variation_mean_l.tolist(),
                                y_lower=q_table_variation_lower_l.tolist(),
                                y_upper=q_table_variation_upper_l.tolist(),
                                name="q variation", y_legend="q variation %")
    # Export Image
    chart.export_as_png(os.path.join(dir, chart_name))
    # Create image Qvariantion-learning_rate:
    chart_name = "learning_rate_q_variation.png"
    chart = TwoLineChart(x_legend="episodes", title="Average q learning "
                         "variation of {} training iterations".
                         format(num_instances))
    chart.add_first_line_chart(x=x, y=lr_mean_l.tolist(),
                               name="learning rate", y_legend="learning rate")
    chart.add_second_line_chart(x=x, y=q_table_variation_mean_l.tolist(),
                                y_lower=q_table_variation_lower_l.tolist(),
                                y_upper=q_table_variation_upper_l.tolist(),
                                name="q variation", y_legend="q variation %")
    # Export Image
    chart.export_as_png(os.path.join(dir, chart_name))
    # Create image reward-epsilon:
    chart_name = "epsilon_reward.png"
    chart = TwoLineChart(x_legend="episodes",
                         title="Average reward of {} training iterations".
                         format(num_instances))
    chart.add_first_line_chart(x=x, y=epsilons_mean_l,
                               name="epsilon", y_legend="epsilon")
    chart.add_second_line_chart(x=x, y=reward_mean_l.tolist(),
                                y_lower=reward_lower_l.tolist(),
                                y_upper=reward_upper_l.tolist(),
                                name="reward", y_legend="reward")
    # Export Image
    chart.export_as_png(os.path.join(dir, chart_name))
    # Create image reward-learning_rate:
    chart_name = "learning_rate_reward.png"
    chart = TwoLineChart(x_legend="episodes",
                         title="Average reward of {} training iterations".
                         format(num_instances))
    chart.add_first_line_chart(x=x, y=lr_mean_l.tolist(),
                               name="learning rate", y_legend="learning rate")
    chart.add_second_line_chart(x=x, y=reward_mean_l.tolist(),
                                y_lower=reward_lower_l.tolist(),
                                y_upper=reward_upper_l.tolist(),
                                name="reward", y_legend="reward")
    # Export Image
    chart.export_as_png(os.path.join(dir, chart_name))

    # Create image reward-q_variation:
    chart_name = "q_variantion_reward_data.png"
    chart = TwoLineChart(title="Average reward of {} training "
                               "iterations".format(num_instances),
                         x_legend="episodes")
    chart.add_first_line_chart(x=x, y=q_table_variation_mean_l.tolist(),
                               y_lower=q_table_variation_lower_l.tolist(),
                               y_upper=q_table_variation_upper_l.tolist(),
                               name="q variation", y_legend="q variation (%)")
    chart.add_second_line_chart(x=x, y=reward_mean_l.tolist(),
                                y_lower=reward_lower_l.tolist(),
                                y_upper=reward_upper_l.tolist(),
                                name="reward", y_legend="reward")
    # Export Image
    chart.export_as_png(os.path.join(dir, chart_name))

    

