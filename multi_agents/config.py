###############
# Exploration #
###############

EXPLORAION_POLICY = "epsilon-greedy"
INITIAL_COLLECT_STEPS = 0
START_EXPLORATION_RATE = 0.5
END_EXPLORATION_RATE = 0.005
FINAL_EXPLORATION_TIMESTEP = 5000

##########
# Models #
##########

# Global parameters shared between models

EPOCHS = 2
BATCH_SIZE = 16_000
MINIBATCH_SIZE = 64  # 32
NUM_MIN_STABLE_TRAINING_EP = 10

# Teammate models
TEAMMATE_MODEL_LAYERS = (
    (48, "relu"),
    (48, "relu")
)

# Model-specific parameters
LEARNING_RATE = 0.00025  # 0.001
DQN_LAYERS = {
    "input": (512, "relu"),
    "hidden": (
        (512, "relu"),
        (512, "relu")
    ),
    "output": (None, "linear")
}
DQN_DISCOUNT_FACTOR = 0.995  # 0.995

###########
# PLASTIC #
###########

BASE_MODEL_DQN = "/home/matias/Desktop/HFO/matias_hfo/models/base/agent_model"
TEAM_EXPERIENCE_BUFFER_FORMAT = "team_learn_buffer.{step}"
DQN_EXPERIENCE_BUFFER_FORMAT = "dqn_learn_buffer.{step}"
MODEL_FILE_FORMAT = "{step}.model"

# PLASTIC_FORMATS:
PLASTIC_MODEL_FORMAT = "plastic.pickle"
DQN_MODEL_FORMAT = "dqn_model.model"
TEAM_MODEL_FORMAT = "team_model.model"
REPLAY_BUFFER_FORMAT = "experience_buffer.pkl"

# Teams:
TEAMS_NAMES = ["aut", "axiom", "cyrus", "gliders", "helios"]  # "agent2d"

# Plastic Models:
ETA = 0.25   # Maximum loss for PLASTIC Belief Updates. Original values
NN_BATCH_SIZE = 750_000
