class Policy:
    """ Encapsulates a team model. """
    
    def __init__(self, team_name):
        self.team_name = team_name
        
        # DQN
        self._dqn = DQN(num_actions=0, num_features=0,
                        learning_rate=config.LEARNING_RATE)
        self._team_model = None
        self._replay_buffer = ReplayBuffer(
            memory_size=config.REPLAY_MEMORY_SIZE)
    
    @property
    def dqn(self) -> DQN:
        return self._dqn
    
    @property
    def team_model(self) -> TeamModel:
        if self._team_model is None:
            raise Exception("Team Model is not trained!")
        else:
            return self._team_model
    
    def train_team_model(self, data: List[Transition]):
        self._team_model = TeamModel.create_model(data)
    
    def model_similarity(self, transition: Transition) -> float:
        """
        Returns the likelihood of the model being the one which the agent is
        interacting with
        """
        return self.team_model.similarity(transition)
    
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
    
    def learn_teammate_nn_model(self, data: List[Transition]):
        return NearestNeighbors(n_neighbors=1, algorithm='auto').fit(data)