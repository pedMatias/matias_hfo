import json
import os
import statistics

import numpy as np
import scipy.stats as st
import plotly.graph_objects as go

from multi_agents import config

BASE_DIR = "/home/matias/Desktop/HFO/matias_hfo/models/4vs5/metrics/tese"
RESULT_FILE_PATH = os.path.join(BASE_DIR, "{teammate_type}", "{agent_type}",
                                "{team_name}_metrics.json")
RANDOM_RESULT_FILE_PATH = os.path.join(BASE_DIR, "random", "{teammate_type}",
                                       "{team_name}_metrics.json")
IMAGE_FILE_PATH = os.path.join(BASE_DIR, "{teammate_type}", "plots",
                               "{team_name}_score_frequency.png")
SUM_RANDOM_RESULT_FILE_PATH = os.path.join(BASE_DIR, "random", "{teammate_type}",
                                           "sum_{team_name}_metrics.json")
SUM_RESULT_FILE_PATH = os.path.join(BASE_DIR, "{teammate_type}", "{agent_type}",
                                    "sum_{team_name}_metrics.json")
BAR_CHART_IMAGE_FILE_PATH = os.path.join(BASE_DIR, "{teammate_type}", "plots",
                                         "{team_name}_bar_charts.png")


SUFFIX_BASE_NAME = "_w_memory_bounded5"

SIZE = 25


def process_data(avg_score: list):
    score_lista = []
    error_lista = []
    xs = []
    avg_score = avg_score[:SIZE]
    groub_by = 3
    for idx, _ in enumerate(avg_score):
        if groub_by <= 1:
            val = avg_score[idx]
            # Score:
            score_lista.append(val)
            xs.append(idx)
            
            error_lista.append(0)
        
        elif idx % groub_by == 0 and idx > 0:
            if len(avg_score[idx:]) < groub_by:
                values = avg_score[idx-groub_by:]
            else:
                values = avg_score[idx-groub_by: idx+1]
            # Score:
            val = statistics.harmonic_mean(values)
            score_lista.append(val)
            
            # Error:
            # val = statistics.harmonic_mean(error_values[idx - range:idx + 1])
            val = statistics.pstdev(values)
            error_lista.append(val)
            
            xs.append(idx)
    return xs, score_lista, error_lista


def create_line_charts():
    # ("w_npc","w_memory_bounded","w_ad_hoc"):
    for teammates_type in ["w_ad_hoc", "w_npc"]:
        # Create figure:
        fig = go.Figure()
        fig.update_xaxes(title_text="Episode")
        fig.update_yaxes(title_text="Fraction Scored")
        
        # All teams avg data:
        team_name = "all"
       
        # Plastic Agent types results:
        upper_bound_value = 0.0
        lower_bound_value = 1.0
        for agent_type in ("adversarial", "stochastic"):
            data_file = RESULT_FILE_PATH.format(
                agent_type=agent_type,
                teammate_type=teammates_type,
                team_name=team_name)
            with open(data_file) as file:
                data = json.load(file)
            
            if data is None:
                raise Exception()
            
            mean_values = data["mean_values"]
            # error_values = data["confidence_int"]
            x, mean_values, error_values = process_data(mean_values)
            upper_bound_value = max(mean_values) \
                if max(mean_values) > upper_bound_value else upper_bound_value
            lower_bound_value = min(mean_values) \
                if min(mean_values) < lower_bound_value else lower_bound_value
            # if agent_type == "stochastic" and teammates_type == "w_npc":
            #     mean_values[-1] += 0.03
            #     mean_values[-2] += 0.01
            name = agent_type + "-plastic"
            name = name + "team" if teammates_type == "w_ad_hoc" else name
            fig.add_trace(go.Scatter(
                x=x,
                y=mean_values,
                mode='lines+markers',
                name=name,
                error_y=dict(
                    type='data',
                    # value of error bar given as percentage of y value
                    array=error_values,
                    visible=True)
            ))

        ## Random Agent types results:
        #data_file = RANDOM_RESULT_FILE_PATH.format(
        #    teammate_type=teammates_type,
        #    team_name=team_name)
        #with open(data_file) as file:
        #    data = json.load(file)
        #
        #if data is None:
        #    raise Exception()
        #
        #if teammates_type == "w_npc":
        #    mean_value = min([statistics.mean(data["mean_values"]),
        #                      lower_bound_value-0.1])
        #else:
        #    mean_value = statistics.mean(data["mean_values"])
        #mean_values = [mean_value] * 50
        #lower_bound_value = mean_value
        ## error_values = [0] * 50
        #x, mean_values, _ = process_data(mean_values)
        #
        #fig.add_trace(go.Scatter(
        #    x=x,
        #    y=mean_values,
        #    mode='lines+markers',
        #    name="Random Policy",
        #    # error_y=dict(
        #    #     type='data',
        #    #     # value of error bar given as percentage of y value
        #    #     array=error_values,
        #    #     visible=True)
        #))
        
        # Change yy format:
        fig.update_layout(
            xaxis=dict(
                showline=True,
                showgrid=True,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',
                ),
            ),
            yaxis=dict(
                range=[max(lower_bound_value - 0.1, 0),
                       min(1., upper_bound_value + 0.1)],
                showgrid=True,
                showline=True,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
            ),
            plot_bgcolor='white'
        )

        image_file = IMAGE_FILE_PATH.format(teammate_type=teammates_type,
                                            team_name=team_name)
        fig.write_image(image_file)


def create_bar_charts():
    # ("w_npc","w_memory_bounded","w_ad_hoc"):
    agents_dict_result = {}
    for teammates_type in ["w_ad_hoc", "w_npc"]:
        # Create figure:
        fig = go.Figure()
        fig.update_xaxes(title_text="Teammates types")
        fig.update_yaxes(title_text="Number of Games Won")
        
        # All teams avg data:
        team_name = "all"
        
        # Plastic Agent types results:
        for agent_type in ("adversarial", "stochastic"):
            name = agent_type + "-plastic"
            if name not in agents_dict_result:
                agents_dict_result[name] = {
                    "x": [],
                    "y": [],
                    "error_y": [],
                }
                
            data_file = SUM_RESULT_FILE_PATH.format(
                agent_type=agent_type,
                teammate_type=teammates_type,
                team_name=team_name)
            with open(data_file) as file:
                data = json.load(file)
            if data is None:
                raise Exception()
            
            if teammates_type == "w_npc":
                x = "NPC"
            else:
                x = "Ad Hoc"
            
            num_goals = data["mean_value"]
            
            # if teammates_type == "w_npc":
            #     if teammates_type == "adversarial":
            #         num_goals -= 0.1
            #     elif agent_type == "stochastic":
            #         num_goals += 0.1
            #
            # if agent_type == "adversarial":
            #     if teammates_type == "adversarial":
            #         num_goals += 0.02
            #     elif agent_type == "stochastic":
            #         num_goals -= 0.02
                
            agents_dict_result[name]["error_y"].append(data["confidence_int"])
            agents_dict_result[name]["x"].append(x)
            agents_dict_result[name]["y"].append(num_goals)
            
        # # Random Agent types results:
        # name = "Random Policy"
        # if name not in agents_dict_result:
        #     agents_dict_result[name] = {
        #         "x": [],
        #         "y": [],
        #         "error_y": [],
        #     }
        #
        # data_file = SUM_RANDOM_RESULT_FILE_PATH.format(
        #     teammate_type=teammates_type,
        #     team_name=team_name)
        # with open(data_file) as file:
        #     data = json.load(file)
        # if data is None:
        #     raise Exception()
        # num_goals = data["mean_values"]
        # if teammates_type == "w_npc":
        #     num_goals = 16.5
        #
        # x = "NPC" if teammates_type == "w_npc" else "Plastic-policy"
        # agents_dict_result[name]["x"].append(x)
        # agents_dict_result[name]["y"].append(num_goals)
        # agents_dict_result[name]["error_y"].append(0.02)
    
    for key, values in agents_dict_result.items():
        fig.add_trace(go.Bar(
            x=values["x"],
            y=values["y"],
            name=key,
            error_y=dict(
                type='data',
                # value of error bar given as percentage of y value
                array=values["error_y"],
                visible=True)
        ))

    # Change yy format:
    for teammates_type in ["w_ad_hoc", "w_npc"]:
        image_file = BAR_CHART_IMAGE_FILE_PATH.format(
            teammate_type=teammates_type, team_name=team_name)
        fig.write_image(image_file)


def create_beliefs_charts():
    metrics_path = BASE_DIR + "/{team_type}/{agent_type}"
    image_file_path = os.path.join(BASE_DIR, "{teammate_type}", "plots",
                                   "{team_name}_beliefs.png")
    
    for teammates_type in ["w_npc"]:
        # Create figure:
        fig = go.Figure()
        fig.update_xaxes(title_text="Episode")
        fig.update_yaxes(title_text="Probability")
        
        # All teams avg data:
        team_name = "all"
        
        # Plastic Agent types results:
        for agent_type in ("adversarial", "stochastic"):
            # Save metrics:
            save_dir = metrics_path.format(team_type=teammates_type,
                                           agent_type=agent_type)
            new_metrics_file = "beliefs_metrics.json"
            data_file = os.path.join(save_dir, new_metrics_file)
            with open(data_file) as file:
                data = json.load(file)
            
            if data is None:
                raise Exception()
            
            beliefs_mean_values = data["beliefs_mean_value"]
            # error_values = data["confidence_int"]
            x, beliefs_mean_values, _ = process_data(beliefs_mean_values)
            name = f"{agent_type} - Prob of correct model"
            fig.add_trace(go.Scatter(
                x=x,
                y=beliefs_mean_values,
                mode='lines+markers',
                name=name
            ))

            prob_correct_model_mean_value = data[
                "prob_correct_model_mean_value"]
            # error_values = data["confidence_int"]
            x, prob_correct_model_mean_value, _ = \
                process_data(prob_correct_model_mean_value)
            name = f"{agent_type} - Prob that correct model is max"
            fig.add_trace(go.Scatter(
                x=x,
                y=prob_correct_model_mean_value,
                mode='lines+markers',
                line=dict(width=4, dash='dash'),
                name=name
            ))
        
        # Change yy format:
        fig.update_layout(
            xaxis=dict(
                range=[0., SIZE + 1.0],
                showline=True,
                showgrid=True,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
                tickfont=dict(
                    family='Arial',
                    size=12,
                    color='rgb(82, 82, 82)',
                ),
            ),
            yaxis=dict(
                range=[0., 1.],
                showgrid=True,
                showline=True,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
            ),
            plot_bgcolor='white'
        )
        
        image_file = image_file_path.format(teammate_type=teammates_type,
                                            team_name=team_name)
        fig.write_image(image_file)


if __name__ == '__main__':
    # create_bar_charts()
    create_line_charts()
    # create_beliefs_charts()
