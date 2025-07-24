import numpy as np

from collections import defaultdict, deque
from tabular_q_agent_parent import Agent

import pickle

class TabularQAgent(Agent):
    def __init__(self, agent_id: int, n_actions: int, n_states: int, config: dict):
        """
        Initialize a Q-learning agent with a Q-table.

        :param agent_id: The ID of the agent.
        :param n_actions: The number of available actions.
        :param n_states: The number of states in the environment.
        :param config: A dictionary containing agent configuration parameters.
        """
        super().__init__(agent_id, n_actions)
        self.n_states              = n_states
        self.initial_q             = config["initial_q"]
        # instead of a big array, use a defaultdict that creates a vector of size n_actions
        self.q                     = {}  # state → np.array of shape (n_actions,)
        self.q_visits              = {}  # state → np.array of ints (n_actions,)
        self.train_freq            = config["train_freq"]
        self.discount              = config["discount"]
        self.learning_rate_mode    = config["learning_rate_mode"]
        self.learning_rate         = config["learning_rate"]
        self.learning_rate_decay   = config["learning_rate_decay"]
        self.exploration           = config["exploration"]
        self.exploration_decay     = config["exploration_decay"]
        self.exploration_min       = config["exploration_min"]
        self.debug                 = config["debug"]
        self.name                  = f"table-q agent {agent_id}"
        self.training_data         = []
        self.training_episodes     = deque(maxlen=self.train_freq)
        self.n_training_episodes   = 0
        self.all_training_episodes = []
        self.cumulative_reward     = 0
        self.cumulative_rewards    = []

    def _ensure_state(self, state):
        """Create Q & visits arrays for a new state if needed."""
        if state not in self.q:
            self.q[state]        = np.ones(self.n_actions) * self.initial_q
            self.q_visits[state] = np.zeros(self.n_actions, dtype=int)

    def start_game(self, is_training: bool):
        """
        Set whether the agent is in training mode and reset cumulative rewards.

        :param do_training: Set training mode of agent.
        """
        super().start_game(is_training)
        self.training_data = []
        self.cumulative_reward = 0


    def act(self, state, actions):
        """
        Select an action using an epsilon-greedy policy.

        :param state: The current state.
        :param actions: List of available actions.
        :return: The selected action.
        """
        # make sure the arrays exist before we read them
        self._ensure_state(state)

        # Explore
        if np.random.uniform(0, 1) < self.exploration and self.is_training:
            action = np.random.choice(actions)
        # Exploit
        else:
            # Disable q-values of illegal actions
            illegal_actions = np.setdiff1d(np.arange(self.n_actions), actions)

            # work on a copy, leave the table intact
            q_row = self.q[state].copy()
            q_row[illegal_actions] = -np.inf          # mask only in the copy

            best = np.flatnonzero(q_row == q_row.max())
            action = np.random.choice(best)

        # Decrease exploration rate
        self.exploration = np.max([self.exploration * (1-self.exploration_decay), self.exploration_min])

        if self.debug:
            print(f"Pick action {action} in state {state} with q-values {self.q[state]}")

        return action

    def update(self, iteration, state, legal_actions, action, reward, done):
        """
        Update the Q-values based on the Q-learning update rule.

        :param iteration: The current iteration number.
        :param state: The current state.
        :param legal_actions: List of legal actions.
        :param action: The selected action.
        :param reward: The observed reward.
        :param done: True if the episode is done, False otherwise.
        """
        super().update(iteration, state, legal_actions, action, reward, done)

        self.cumulative_reward += reward
        if self.is_training:
            self.training_data.append([iteration, state, legal_actions, action, reward, done])

    def final_update(self, reward : float):
        """
        Update the Q-values at the end of an episode. Sets the last training input to done, adds final rewards, validates training data and refreshes training buffer.

        :param reward: The observed reward.
        """
        super().final_update(reward)

        self.cumulative_reward += reward

        if self.is_training:
            self.training_data[-1][self.DONE]    = True
            self.training_data[-1][self.REWARD] += reward

            self.validate_training_data()

            self.training_episodes.append(self.training_data)
            self.training_data = []

            self.cumulative_rewards.append(self.cumulative_reward)
            self.cumulative_reward = 0

    def validate_training_data(self):
        """
        Validate the integrity of training data, checking for missing iterations and incomplete episodes.
        """

        for i in range(len(self.training_data) - 1):
            # Validate iteration number
            i1 = self.training_data[i][self.ITERATION]
            i2 = self.training_data[i+1][self.ITERATION]
            if (i1 + 1 != i2):
                raise ValueError(f"Missing iteration between iterations {i1} and {i2} in training data")

    def train(self, num_episodes=None):
        """
        Train the agent by updating Q-values based on the collected training data.

        Parameters:
        - num_episodes (int, optional): Number of episodes to use from the training_episodes buffer.
        If None, use all available episodes.
        """
        if not self.is_training or not self.training_episodes:
            return


        # Determine how many episodes to train on
        if num_episodes is None:
            episodes_to_use    = self.training_episodes
            remaining_episodes = []
        else:
            episodes_to_use    = self.training_episodes[:num_episodes]
            remaining_episodes = self.training_episodes[num_episodes:]


        next_max = None

        next_iteration = 0
        next_state = 0

        for i, data in enumerate(episodes_to_use):
            for iteration, state, legal_actions, action, reward, done in reversed(data):

                # make sure the arrays exist before we read them
                self._ensure_state(state)

                # Happens in the round the agent dies
                if action == None:
                    action = 0

                # Increment visit count
                self.q_visits[state][action] += 1

                if self.learning_rate_mode == "adaptive":
                    # Compute adaptive learning rate
                    alpha = max(0.001, min(self.learning_rate, 1.0 / (1 + self.q_visits[state][action])))
                elif self.learning_rate_mode == "fixed":
                    alpha = self.learning_rate
                else:
                    raise ValueError("Unknown learning_rate_mode")

                # Q-learning update rule
                if done:
                    self.q[state][action] = reward
                else:
                    self.q[state][action] += alpha * (reward + self.discount * next_max - self.q[state][action])

                next_max = np.max(self.q[state])

            self.n_training_episodes += 1

        # move to buffer for all episodes
        self.all_training_episodes += self.training_episodes

        # Move only used episodes to buffer
        self.all_training_episodes += episodes_to_use

        # Keep only remaining episodes
        self.training_episodes = remaining_episodes


    def save_transitions(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.all_training_episodes), f)

    def load_transitions(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            self.training_episodes = pickle.load(f)
