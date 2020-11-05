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

LEARNING_RATE = 0.001
REPLAY_MIN_BATCH = 32
REPLAY_MEMORY_SIZE = 15000

# Teammate models
TEAMMATE_MODEL_LAYERS = (
    (48, "relu"),
    (48, "relu")
)

# Model-specific parameters
DQN_LAYERS = (
    (256, "relu"),
    (256, "relu")
)
DQN_DISCOUNT_FACTOR = 0.95

###########
# PLASTIC #
###########

ETA = 0.25   # Maximum loss for PLASTIC Belief Updates. Original values
