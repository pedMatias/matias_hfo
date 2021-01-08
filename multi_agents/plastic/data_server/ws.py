#!flask/bin/python
import os
import json

from flask import Flask, request, jsonify, abort
import numpy as np
import keras
import tensorflow as tf

from multi_agents import config
from multi_agents.plastic.policy import Policy
from multi_agents.dqn_agent.replay_buffer import Transition


MODEL_DIR = "/home/matias/Desktop/HFO/matias_hfo/models/2vs2"

app = Flask(__name__)


def load_plastic_policies(dir_path: str, team_names: list):
    print("[LOAD POLICIES] Started!")
    keras.backend.set_session(session)
    if not os.path.isdir(dir_path):
        print(f"[load_plastic_models] Dir not found {dir_path};")
        raise NotADirectoryError(dir_path)
    policies_dict = dict()
    for team_name in team_names:
        if not os.path.isdir(os.path.join(dir_path, team_name)):
            print(f":: Can not find team {team_name}!\n".upper())
        else:
            policy = Policy.load(team_name=team_name, base_dir=dir_path)
            policies_dict[team_name] = policy
            print(f":: Found Policy {team_name};")
    print("[LOAD POLICIES] DONE!")
    return policies_dict


@app.route('/predict', methods=['POST'])
def predict_task():
    keras.backend.set_session(session)
    if not request.json:
        predicted_array = None
        abort(400)
    else:
        data = json.loads(request.json)
        assert "state" in data.keys()
        assert "team_name" in data.keys()
        team_name = data["team_name"]
        
        state = np.array(data["state"]).astype(np.float32)
        with session.as_default():
            with session.graph.as_default():
                predicted_array = policies[team_name].dqn.predict(state)[0]
    return jsonify(predicted_array.tolist()), 200


@app.route('/similarity', methods=['POST'])
def similarity_task():
    if not request.json:
        sim_dict = None
        abort(400)
    else:
        data = json.loads(request.json)
        assert "state" in data.keys()
        assert "next_state" in data.keys()
        state = np.array(data["state"]).astype(np.float32)
        next_state = np.array(data["next_state"]).astype(np.float32)
        transition = Transition(obs=state, act=0, reward=0, new_obs=next_state,
                                done=False)
        
        sim_dict = {}
        for policy_name, policy in policies.items():
            sim_dict[policy_name] = policy.model_similarity(transition)
    return jsonify(sim_dict), 200


@app.route('/team_names', methods=['GET'])
def get_team_names():
    team_names = [policy.team_name for policy in policies]
    return jsonify(team_names), 200


if __name__ == '__main__':
    session = tf.Session()
    policies = load_plastic_policies(MODEL_DIR, config.TEAMS_NAMES)
    app.run(debug=False, use_reloader=False, threaded=True)
