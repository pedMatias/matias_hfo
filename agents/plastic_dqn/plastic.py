import numpy as np

from agents.plastic_dqn import config
from agents.plastic_dqn.dqn import DQN
from agents.plastic_dqn.replay_buffer import ReplayBuffer


class Policy:
    """ Encapsulates a team model. """
    
    def __init__(self, directory, team_name, num_teammates):
        self.directory = directory
        self.name = team_name
        self.num_teammates = num_teammates
        self.num_agents = self.num_teammates + 1
        
        # DQN
        self.dqn = DQN(num_actions=0, num_features=0,
                       learning_rate=config.LEARNING_RATE)
        self.teammates_model = self.setup_models()
        self.replay_buffer = ReplayBuffer(memory_size=config.REPLAY_MEMORY_SIZE)
    
    def likelihood_given_actions(self, state, teammates_actions):
        """
        Returns the likelihood of the model being the one which the agent is interacting with
        """
        policies = self.policies(state)
        probabilities = []
        for teammate_id, action in enumerate(teammates_actions):
            policy = policies[teammate_id]
            probability = policy[action]
            probabilities.append(probability)
        probabilities = np.array(probabilities)
        return np.multiply.reduce(probabilities)
    
    def policies(self, state):
        """
        Given a state, returns the teammates's policies
        """
        return self.teammates_model.policies(state)
    
    def simulate_teammates_actions(self, state):
        """
        Given a state, predicts a possible set of teammate actions,
        Given their policies
        """
        policies = self.policies(state)
        num_action = 4
        indices = [np.random.choice(range(num_action), p=pi) for pi in
                   policies]
        return indices


class NearestNeighboursModel:
    pass


def learn_about_prior_teammate(teammate_name: str) -> (Policy, NearestNeighboursModel):
    """
    Plays with the teammates for N timesteps.
    At each timestep collects (s, a, r, s');
    Learns a policy P using DQN
    Learns a nearest neighbors model M
    :@return (P, N)
    """
    raise NotImplementedError

def get_policy(nn_model):
    pass

def normalize(dict):
    pass

def update_beliefs(behaviour_dist: dict, s: np.ndarray, a: int, n: int):
    """
    @param behaviour_dist:probability distr. over possible teammate behaviors
    @param s: the previous environment state
    @param a: previously chosen action
    @param n: bounds the maximum allowed loss
    @return BehaviorDistr: updated probability distr
    """
    for nn_model in behaviour_dist.keys():
        policy = get_policy(nn_model)
        loss = 1 - policy.predict(s, a)  # loss = 1 −P(a|m, s)
        behaviour_dist[nn_model] *= (1 - n*loss)  # BehaviorDistr(m)∗ = (1 − ηloss)
    behaviour_dist = normalize(behaviour_dist)
    return behaviour_dist


def select_action(behaviour_distr: dict, s: np.ndarray) -> int:
    """
    @param behaviour_distr: probability distr. over possible teammate behaviors
    @param s: the current environment state
    @return: the best action for the agent to take
    """
    """
    (π,m) = argmax BehaviorDistr
    select most likely policy
    a = π(s)
    return a
    """
    pass


def plastic(prior_teammates: list, hand_coded_knowledge: list,
            behavior_prior):
    prior_knowledge = hand_coded_knowledge
    for t in prior_teammates:
        prior_knowledge += learn_about_prior_teammate(t)
    behavior_distr = behavior_prior(prior_knowledge)
    
    s = None
    while not terminal(s):
        a = select_action(behavior_distr, s)
        r, new_s = step(a)
        behavior_distr = update_beliefs(behavior_distr, s, a)