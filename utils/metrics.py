import json
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create random data with numpy
import numpy as np
np.random.seed(1)

from utils.aux_functions import get_mean_value_list_by_range

STATES_MAP = {"TOP LEFT": list(range(0, 120, 6)),
              "TOP RIGHT": list(range(1, 120, 6)),
              "MID LEFT": list(range(2, 120, 6)),
              "MID RIGHT": list(range(3, 120, 6)),
              "BOTTOM LEFT": list(range(4, 120, 6)),
              "BOTTOM RIGHT": list(range(5, 120, 6))}


class Plot:
    def __init__(self, title="", x_legend="", y_legend=""):
        self.figure = go.Figure()
        self.figure.update_layout(title=title,
                                  xaxis_title=x_legend,
                                  yaxis_title=y_legend)
    
    def add_line_chart(self, x: list, y:list, name: str, type: str = None):
        """
        @param x: x axis data
        @param y: y axis data
        @param name: line name
        @param type: options include 'dash', 'dot', and 'dashdot'
        @return:
        """
        line = dict(dash=type)
        trace = go.Scatter(x=x, y=y, name=name, line=line)
        self.figure.add_trace(trace)
    
    def export_as_png(self, file_name: str):
        self.figure.write_image(file_name)


class BarChart(Plot):
    def add_bar(self, x: list, y: list, name: str):
        """
        @param x: x axis data
        @param y: y axis data
        @param name: line name
        @return:
        """
        trace = go.Bar(x=x, y=y, name=name)
        self.figure.add_trace(trace)
        self.figure.update_layout(barmode='group')


class LineChart(Plot):
    def add_line_chart(self, x: list, y: list, name: str, type: str = None,
                       y_upper: list = None, y_lower: list = None):
        """
        @param x: x axis data
        @param y: y axis data
        @param name: line name
        @param type: options include 'dash', 'dot', and 'dashdot'
        @return:
        """
        line = dict(dash=type)
        self.figure.add_trace(go.Scatter(x=x, y=y, name=name, line=line))
        if y_upper and y_lower:
            x_rev = x[::-1]
            y_lower = y_lower[::-1]
            self.figure.add_trace(go.Scatter(
                x=x + x_rev,
                y=y_upper + y_lower,
                fill='toself',
                #fillcolor='rgba(0,100,80,0.2)',
                #line_color='rgba(255,255,255,0)',
                showlegend=False,
                name=name,
            ))


class TwoLineChart(Plot):
    def __init__(self, title="", x_legend="", y_legend=""):
        self.figure = make_subplots(specs=[[{"secondary_y": True}]])
        self.figure.update_layout(title_text=title)
        self.figure.update_xaxes(title_text=x_legend)
    
    def add_first_line_chart(self, x: list, y: list, name: str, y_legend: str,
                             y_upper: list = None, y_lower: list = None):
        """
        @param x: x axis data
        @param y: y axis data
        @param name: line name
        @param y_legend:
        @return:
        """
        self.figure.add_trace(
            go.Scatter(x=x, y=y, name=name, line_color='rgba(25,0,250,1)'),
            secondary_y=False,
        )
        self.figure.update_yaxes(title_text=y_legend, secondary_y=False)
        if y_upper is not None and y_lower is not None:
            x_rev = x[::-1]
            y_lower = y_lower[::-1]
            self.figure.add_trace(
                go.Scatter(
                    x=x + x_rev,
                    y=y_upper + y_lower,
                    fill='toself',
                    fillcolor='rgba(25,0,250,0.2)',
                    line_color='rgba(255,255,255,0)',
                    showlegend=False,
                    name=name
                ),
                secondary_y=False,
            )
    
    def add_second_line_chart(self, x: list, y: list, name: str,
                              y_legend: str, y_upper: object = None,
                              y_lower: object = None):
        """
        @param x: x axis data
        @param y: y axis data
        @param name: line name
        @param y_legend:
        @return:
        """
        self.figure.add_trace(
            go.Scatter(x=x, y=y, name=name, line_color='rgba(250,0,0,1)'),
            secondary_y=True,
        )
        self.figure.update_yaxes(title_text=y_legend,
                                 secondary_y=True)
        if y_upper is not None and y_lower is not None:
            x_rev = x[::-1]
            y_lower = y_lower[::-1]
            self.figure.add_trace(
                go.Scatter(
                    x=x + x_rev,
                    y=y_upper + y_lower,
                    fill='toself',
                    fillcolor='rgba(250,0,0,0.2)',
                    line_color='rgba(255,255,255,0)',
                    showlegend=False,
                    name=name),
                secondary_y=True,
            )


class HeatMapPlot:
    def __init__(self, data: list, x_labels: list, y_labels: list):
        self.figure = go.Figure(data=go.Heatmap(z=data, x=x_labels, y=y_labels))

    def export_as_png(self, file_name: str):
        self.figure.write_image(file_name)


def json_to_metrics(dir: str, json_file: str):
    with open(json_file) as fp:
        data = json.loads(fp.read())
    
    mode: str = data.get("mode")
    num_trained_episodes: list = data.get("num_trained_episodes")
    epsilons: list = data.get("epsilons")
    score: list = data.get("score")
    q_variation: list = data.get("q_variation")
    actions_label: list = data.get("actions_label")
    reward: list = data.get("reward")
    visited_states_counter: list = data.get("visited_states_counter")
    
    # Pre-process data:
    if reward:
        reward = get_mean_value_list_by_range(reward, 50)
    if q_variation:
        q_variation = get_mean_value_list_by_range(q_variation, 50)
    print("epsilons", epsilons)
    print("q_variation", q_variation)
    print("actions_label", actions_label)
    print("reward", reward)
    print("visited_states_counter", visited_states_counter)
    
    # Epsilon-Q_variation:
    if epsilons and q_variation:
        chart_name = "{}_epsilon_q_variation.png".format(mode)
        episodes = list(range(1, len(epsilons)))
        chart = TwoLineChart(x_legend="episodes")
        chart.add_first_line_chart(x=episodes, y=epsilons,
                                   name="epsilon", y_legend="epsilon")
        chart.add_second_line_chart(x=episodes, y=q_variation,
                                    name="q variation", y_legend="q variation")
        chart.export_as_png(os.path.join(dir, chart_name))
    
    # Epsilon-Reward:
    if epsilons and reward:
        chart_name = "{}_epsilon_reward.png".format(mode)
        episodes = list(range(1, len(epsilons)))
        chart = TwoLineChart(x_legend="episodes")
        chart.add_first_line_chart(x=episodes, y=epsilons,
                                   name="epsilon", y_legend="epsilon")
        chart.add_second_line_chart(x=episodes, y=reward,
                                    name="reward", y_legend="reward")
        chart.export_as_png(os.path.join(dir, chart_name))
    # Q_variation-Reward:
    if q_variation and reward:
        chart_name = "{}_q_variation_reward.png".format(mode)
        episodes = list(range(1, len(q_variation)))
        chart = TwoLineChart(x_legend="episodes")
        chart.add_first_line_chart(x=episodes, y=q_variation,
                                   name="q_variation", y_legend="q_variation")
        chart.add_second_line_chart(x=episodes, y=reward,
                                    name="reward", y_legend="reward")
        chart.export_as_png(os.path.join(dir, chart_name))
    # Score:
    if score:
        chart_name = "{}_score.png".format(mode)
        chart = BarChart(x_legend="episodes")
        chart.add_bar(x=num_trained_episodes, y=score, name="score")
        chart.export_as_png(os.path.join(dir, chart_name))
    # Visited States Counter - Actions Label:
    if visited_states_counter and actions_label:
        chart_name = "{}_visited_states_heat_map.png".format(mode)
        y_label = []
        new_data = []
        for key, values in STATES_MAP.items():
            aux_list = [visited_states_counter[i] for i in values]
            array = np.array(aux_list)
            res = np.sum(array, 0)
            new_data.append(res.tolist())
            y_label.append(key)
        c4 = HeatMapPlot(data=new_data, x_labels=actions_label,
                         y_labels=y_label)
        c4.export_as_png(os.path.join(dir, chart_name))
        

