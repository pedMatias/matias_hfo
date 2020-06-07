import numpy as np


class QLearner:
    def __init__(self, num_states: int, num_actions: int, learning_rate=0.10,
                 discount_factor=0.9, epsilon=1, epsilon_dec=0.996,
                 epsilon_end=0.01, save_file="q_table"):
        self.num_states = num_states
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.learn_rate = learning_rate
        self.discount = discount_factor
        self.save_file = save_file
        self.q_table = np.zeros((num_states, num_actions))

    def update_q(self, old_state_idx: int, action: int, reward: int,
               new_state_idx: int, terminal_state: bool = False):
        self.q_table[old_state_idx][action] *= (1 - self.learn_rate)
        if terminal_state:
            self.q_table[old_state_idx][action] += self.learn_rate * reward
        else:
            self.q_table[old_state_idx][action] += self.learn_rate * \
                (reward + self.discount * np.amax(self.q_table[new_state_idx]))

    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_dec \
            if self.epsilon > self.epsilon_end else self.epsilon_end

    def choose_action(self, state_idx: int):
        if np.random.random() < self.epsilon:  # Explore
            random_action = np.random.randint(0, self.num_actions)
            return random_action
        else:  # Exploit
            return self.exploit_action(state_idx)

    def exploit_action(self, state_idx: int):
        # If multiple equal q-values, pick randomly
        max_list = np.where(self.q_table[state_idx]
                            == self.q_table[state_idx].max())
        if len(max_list[0]) > 1:
            action = np.random.randint(0, len(max_list[0]))
            return action
        return np.argmax(self.q_table[state_idx])

    def save_q_table(self, other_file=None):
        if other_file:
            np.save(other_file, self.q_table)
        else:
            np.save(self.save_file, self.q_table)

    def load_q_table(self, file):
        try:
            self.q_table = np.load(file)
        except Exception as e:
            print("Failed to load input file - " + str(e))
            self.q_table = np.zeros((self.num_states, self.num_actions))
