
import os
import random
import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint  # Import ModelCheckpoint callback
from tensorflow.keras import layers, models

from q_agent_parent import Agent
from q_prioritised_experience_replay import PrioritizedReplayBuffer
from helpers import TransitionFields

# Define a directory to save checkpoints and logs
checkpoint_dir = 'checkpoints'
log_dir = 'logs'

# Create directories if they don't exist
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Base class for sampling buffer (either uniform buffer or prioritised experience replay buffer)
class ReplaySampler:
    def sample(self, buffer, batch_size):
        raise NotImplementedError

    def add(self, buffer, transition):
        raise NotImplementedError

    def update_priorities(self, buffer, idxs, td_errors):
        pass  # No-op for uniform replay

class UniformReplaySampler(ReplaySampler):
    def sample(self, buffer, batch_size):
        minibatch = random.sample(buffer, batch_size)
        idxs = None
        weights = np.ones(batch_size, dtype=np.float32)  # uniform weights
        return minibatch, idxs, weights

    def add(self, buffer, transition):
        buffer.append(transition)

class PrioritizedReplaySampler(ReplaySampler):
    td_error_list = []

    def sample(self, buffer, batch_size):
        minibatch, idxs, weights = buffer.sample(batch_size)
        return minibatch, idxs, weights

    def update_priorities(self, buffer, idxs, td_errors):
        buffer.update_priorities(idxs, td_errors)
        self.td_error_list.append(td_errors)

    def add(self, buffer, transition):
        buffer.add(transition)

class DeepQAgent(Agent):
    # encoding of feature vector
    ENCODING_BINARY   = 0 # convert tictactoe board that can be considered as 9-digit base 3-number to base 2-number
    ENCODING_TUTORIAL = 1 # use one-hot encoding and convert board to three boolean arrays of size 9 respectively indicating empty fields, crosses and naughts
    ENCODING_SYMMETRY = 2 # encode the board modulo its symmetries
    INPUT_ENCODINGS = {
        "encoding_binary": ENCODING_BINARY,
        "encoding_tutorial": ENCODING_TUTORIAL,
        "encoding_symmetry": ENCODING_SYMMETRY
    }
    LARGE_NEGATIVE_NUMBER = -1e6

    TERMINAL_STATE_ID = -1

    def __init__(self, agent_id, n_actions, n_states, config):
        """
        Initialize a Deep Q-Network (DQN) agent for reinforcement learning.

        :param agent_id:            The ID of the agent.
        :param n_actions:           The number of available actions.
        :param n_states:            The number of states in the environment.
        :param config:              A dictionary containing agent configuration parameters.
                                    - 'board_size': The size of the game board.
                                    - 'n_episode': The total number of episodes for training.
                                    - 'n_eval': The frequency of evaluation episodes.
                                    - 'eval_freq': The evaluation frequency in episodes.
                                    - 'discount': The discount factor (gamma).
                                    - 'learning_rate': The learning rate (alpha).
                                    - 'learning_rate_decay': The learning rate decay factor.
                                    - 'exploration': The exploration rate (epsilon).
                                    - 'exploration_decay': The exploration rate decay factor.
                                    - 'exploration_min': Minimum exploration rate.
                                    - 'batch_size': The batch size for training.
                                    - 'replay_buffer_size': The size of the replay buffer.
                                    - 'target_update_tau':  The weighting between online and target network for updating the target network.
        """
        super().__init__(agent_id, n_actions)
        self.n_states            = n_states

        self.name                = f"deep-q agent {agent_id}"
        self.board_size          = config["board_size"]
        self.n_episode           = config["n_episode"]
        self.n_eval              = config["n_eval"]
        self.eval_freq           = config["eval_freq"]
        self.grad_steps          = config["grad_steps"]
        self.discount            = config["discount"]
        self.learning_rate       = config["learning_rate"]
        self.learning_rate_decay = config["learning_rate_decay"]
        self.exploration         = config["exploration"]
        self.exploration_decay   = config["exploration_decay"]
        self.exploration_min     = config["exploration_min"]
        self.batch_size          = config["batch_size"]
        self.replay_buffer_size  = config["replay_buffer_size"]
        self.replay_buffer_min   = config["replay_buffer_min"]
        self.target_update_mode  = config["target_update_mode"]
        self.target_update_freq  = config["target_update_freq"]
        self.target_update_tau   = config["target_update_tau"]
        self.debug               = config["debug"]

        self.q_visits            = np.zeros((n_states, n_actions))

        self.training_round      = 0
        self.training_data       = []
        self.training_log        = []

        self.enable_double_dqn   = config.get("enable_double_dqn", False)
        if self.enable_double_dqn:
            print("Double DQN enabled!")
        self.enable_prio_exp_rep = config.get("enable_prioritised_experience_replay", False)
        if self.enable_prio_exp_rep:
            print("Prioritised Experience Replay enabled!")

        if not self.enable_prio_exp_rep:
            self.replay_buffer  = deque(maxlen=self.replay_buffer_size)
            self.replay_sampler = UniformReplaySampler()
        else:
            self.replay_buffer  = PrioritizedReplayBuffer(
                capacity=self.replay_buffer_size,
                alpha=config.get("prb_alpha", 0.6),
                beta0=config.get("prb_beta0", 0.4),
                # anneal beta to 1 over the course of the training, in practive, we reach 1 a little sooner than at the end because of buffer filling up in the beginning
                beta_steps=config.get("prb_beta_steps", self.n_episode),
                epsilon=config.get("prb_epsilon", 1e-6),
            )

            self.replay_sampler = PrioritizedReplaySampler()


        self._input_cache       = {}

        # choose encoding of feature vector
        self.board_encoding      = self.INPUT_ENCODINGS[config["board_encoding"]]
        self.input_shape         = (self.state_to_input(1).shape[1],)

        # Models need to be defined and compiled in derived classes
        self.online_model        = None
        self.target_model        = None


    def get_q(self):
        """
        Return the approximated Q-table as a NumPy array of shape (n_states, n_actions),
        by querying the target model for all states in the environment.

        :return: A NumPy array representing Q-values for all states.
        """
        q_values = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

        for state in range(self.n_states):
            s_tensor = self.state_to_input(state)         # shape: (1, input_dim)
            q_pred = self.target_model(s_tensor, training=False)  # shape: (1, n_actions)
            q_values[state] = q_pred.numpy().squeeze()

        return q_values



    def update(self, iteration, state, legal_actions, action, reward, done):
        """
        Update the Q-values based on the Q-learning update rule.

        :param iteration:      The current iteration.
        :param state:          The current state.
        :param legal_actions:  List of legal actions.
        :param action:         The selected action.
        :param reward:         The observed reward.
        :param done:           Boolean indicating whether the episode is done.
        """
        super().update(iteration, state, legal_actions, action, reward, done)
        if self.is_training:
            self.training_data.append([iteration, state, legal_actions, action, reward, done])


    def final_update(self, reward):
        """
        Update the training data after the final step of an episode.

        :param reward: The final observed reward.
        """
        super().final_update(reward)

        if self.is_training:
            self.training_data[-1][self.DONE]    = True
            self.training_data[-1][self.REWARD] += reward

            self.validate_training_data()
            self.move_training_data_to_replay_buffer()

            # Decrease exploration rate
            self.exploration = np.max([self.exploration * (1-self.exploration_decay), self.exploration_min])

    def encode_tutorial(self, state: int) -> np.ndarray:
        """One-hot encode in tutorial mode (3×9 vector)."""
        base_arr = decimal_to_base(state, base=3, padding=9)
        rep = np.zeros(3 * 9, dtype=int)
        for i, v in enumerate(base_arr):
            rep[i + v * 9] = 1
        return rep

    def encode_binary(self, state: int) -> np.ndarray:
        """Binary encode the state id into a 15-bit vector."""
        return decimal_to_base(state, base=2, padding=15)


    def state_to_input(self, state):
        """
        Convert the state into an input representation suitable for the neural network.

        :param state: The current state.
        :return:      The input representation of the state.
        """

        # Check cache
        if state in self._input_cache:
            return self._input_cache[state]

        if state == self.TERMINAL_STATE_ID:
           representation = np.zeros(self.input_shape, dtype=np.float32)
        elif self.board_encoding == self.ENCODING_TUTORIAL:
            representation = self.encode_tutorial(state)
        elif self.board_encoding == self.ENCODING_BINARY:
            representation = self.encode_binary(state)
        else:
            raise ValueError("Unsupported input mode")

        # force the dtype here
        tensor = tf.convert_to_tensor(
            representation.reshape(1, -1),
            dtype=tf.float32
        )

        self._input_cache[state] = tensor
        return tensor

    def validate_training_data(self):
        """
        Validate the integrity of training data, checking for missing iterations and incomplete episodes.
        """
        # Check integrity of training data
        if self.training_data[-1][self.DONE] is not True:
            raise ValueError("Last training datum not done")

        # Validate iteration number
        for i in range(len(self.training_data) - 1):
            i1 = self.training_data[i  ][self.ITERATION]
            i2 = self.training_data[i+1][self.ITERATION]
            if (i1 + 1 != i2):
                raise ValueError(f"Missing iteration between iterations {i1} and {i2} in training data")



    def move_training_data_to_replay_buffer(self):
        """
        Move training data to the replay buffer, connecting states with their subsequent states.
        """

        # Connect state with next state and move to replay buffer
        for i in range(len(self.training_data)):
            iteration, state, legal_actions, action, reward, done = self.training_data[i]
            if not done:
                next_state          = self.training_data[i+1][self.STATE]
                next_legal_actions  = self.training_data[i+1][self.LEGAL_ACTIONS]
            else:
                next_state          = self.TERMINAL_STATE_ID
                next_legal_actions  = [] # no legal moves after terminal
            self.replay_sampler.add(self.replay_buffer, [state, legal_actions, action, next_state, next_legal_actions, reward, done])

        self.training_data = []

    def minibatch_to_arrays(self, minibatch):
        """
        Convert a minibatch of experiences into arrays for training.

        :param minibatch: A minibatch of experiences.
        :return:          Tensors containing states, actions, next_states, rewards, and not_terminal flags.
        """
        B = len(minibatch)

        # 1) Build the legal-action and scalar arrays in NumPy
        legal_actions       = np.zeros((B,  self.n_actions  ), dtype=np.float32)
        actions             = np.zeros( B,                     dtype=np.int32  )
        next_legal_actions  = np.zeros((B,  self.n_actions),   dtype=np.float32)
        rewards             = np.zeros( B,                     dtype=np.float32)
        not_terminal        = np.zeros( B,                     dtype=np.float32)

        # 2) Build lists of per-sample state tensors
        state_tensors      = []
        next_state_tensors = []

        for i, (s, l, a, s_, l_, r, d) in enumerate(minibatch):
            legal_actions [i, l]       = 1
            actions       [i]          = a
            next_legal_actions [i, l_] = 1
            rewards       [i]          = r
            not_terminal  [i]          = 1 - d

            # encode to whatever shape your agent needs
            s      = self.state_to_input(s)   # tf.Tensor shape [1, …]
            s_next = self.state_to_input(s_)  # tf.Tensor shape [1, …]

            state_tensors.append(s)
            next_state_tensors.append(s_next)

        # 3) Concatenate into batch tensors [B, …]
        states      = tf.concat(state_tensors,      axis=0)  # [B, shape...]
        next_states = tf.concat(next_state_tensors, axis=0)  # [B, shape...]

        # return tensors
        return (
            states,
            tf.convert_to_tensor(legal_actions),
            tf.convert_to_tensor(actions),
            next_states,
            tf.convert_to_tensor(next_legal_actions),
            tf.convert_to_tensor(rewards),
            tf.convert_to_tensor(not_terminal)
        )

    # 1) A graph fn that takes (state_tensor, legal_action_indices) → action_index
    # Without this , TensorFlow goes through its Python‐level traceback filtering machinery on every call to figure out which frames to show you if an exception happens.
    @tf.function(
      input_signature=[
        tf.TensorSpec(shape=[1, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None],   dtype=tf.int32),
      ]
    )
    def _graph_act(self, s, legal_idxs):
        # get q-values
        q = self.online_model(s, training=False)             # shape [1, n_actions]
        # build a mask inside TF
        mask = tf.scatter_nd(
            tf.expand_dims(legal_idxs, 1),                   # [[i0], [i1], …]
            tf.ones_like(legal_idxs, dtype=tf.float32),      # [1,1,…]
            [self.n_actions]                                 # output shape
        )                                                     # shape [n_actions]
        mask = tf.reshape(mask, [1, -1])                     # [1, n_actions]
        # apply mask + LARGE_NEGATIVE_NUMBER trick
        neg_inf = tf.constant(self.LARGE_NEGATIVE_NUMBER, tf.float32)
        masked_q = mask * q + (1 - mask) * neg_inf           # still [1, n_actions]
        # pick best
        return tf.argmax(masked_q, axis=1)[0]                # a scalar tf.Tensor[int32]


    def act(self, state, actions):
        """
        Select an action using an epsilon-greedy policy.

        :param state:   The current state.
        :param actions: List of available actions.
        :return:        The selected action.
        """

        # explore
        if np.random.uniform(0, 1) < self.exploration and self.is_training:
            action = np.random.choice(actions)
        # exploit
        else:
            s = self.state_to_input(state)
            # one synchronous graph call, no py-side masking or .numpy() inside TF internals:
            action = int(self._graph_act(s, tf.constant(actions, tf.int32)))

        return action


    def update_target_weights(self):
        """
        Update the weights of the target network according to
        target_weight = (1 - tau) * target_weight + tau * online_weight for soft update
        and
        target_weight = online_weight for hard update
        """

        if self.target_update_mode == "soft":
            online_weights = self.online_model.get_weights()
            target_weights = self.target_model.get_weights()

            new_target_weights = [
                (1 - self.target_update_tau) * target_weight + self.target_update_tau * online_weight
                for online_weight, target_weight in zip(online_weights, target_weights)
            ]
            self.target_model.set_weights(new_target_weights)

        else:
            if self.training_round % self.target_update_freq == 0:
                self.target_model.set_weights(self.online_model.get_weights())




    def train(self):
        """
        Train the agent's Q-network using experiences from the replay buffer.
        """

        if len(self.replay_buffer) < self.replay_buffer_min:
            return                          # still warming up

        # Sample a random minibatch from the replay replay_buffer
        for gradient_step in range(self.grad_steps):

            # ───────── 1)  sample  ─────────
            minibatch, idxs, weights = self.replay_sampler.sample(
                self.replay_buffer, self.batch_size
            )

            # bookkeeping (optional – still NumPy friendly)
            for (s, _, a, _, _, _, _) in minibatch:
                self.q_visits[s][a] += 1

            # unpack as **tensors** (dtype=float32 unless noted)
            (states, legal_s, actions, next_states,
            next_legal_s, rewards, not_terminal) = self.minibatch_to_arrays(minibatch)

            # ───────── 2)  forward pass ─────────
            q_current = self.online_model(states, training=False)              # (B, A)
            q_next_t  = self.target_model(next_states, training=False)         # (B, A), used to evaluate Q_target
            q_next_o  = self.online_model(next_states, training=False) if self.enable_double_dqn else q_next_t # (B, A), used to evaluate best action

            # ───────── 3)  mask + argmax on next-state ─────────
            masked_next = tf.where(next_legal_s > 0, q_next_o, self.LARGE_NEGATIVE_NUMBER)      # (B, A)
            best_actions = tf.argmax(masked_next, axis=1, output_type=tf.int32)  # (B,)

            # if **all** actions were illegal in a row, give q=0
            has_legal = tf.reduce_any(next_legal_s > 0, axis=1)                # (B,)
            batch_ids = tf.range(tf.shape(states)[0], dtype=tf.int32)
            gather_nd = tf.stack([batch_ids, best_actions], axis=1)            # (B,2)

            next_q_max = tf.where(
                has_legal,
                tf.gather_nd(q_next_t, gather_nd),     # Q_target(s′, a*)
                tf.zeros_like(rewards)                 # no legal → 0
            )                                          # shape (B,)

            # ───────── 4)  Bellman target ─────────
            true_targets = rewards + not_terminal * self.discount * next_q_max  # (B,)

            # ───────── 5)  TD-error for PER ─────────)   # |δ
            cur_q_sa   = tf.gather_nd(q_current, tf.stack([batch_ids, actions], axis=1))             # (B,)
            td_errors  = true_targets - cur_q_sa                                                     # (B,)
            self.replay_sampler.update_priorities(
                self.replay_buffer, idxs, td_errors.numpy())   # |δ|+ε recommended

            # ───────── 6)  replace Q(s,a) by target in a tensor-safe way ─────────
            q_target_batch = tf.tensor_scatter_nd_update(
                q_current,                                           # base tensor
                tf.stack([batch_ids, actions], axis=1),              # indices of (s,a)
                true_targets                                         # new values
            )                                                        # still (B, A)

            # ───────── 7)  SGD step ─────────
            sample_weights = tf.convert_to_tensor(weights, dtype=tf.float32)

            loss = self.online_model.train_on_batch(
                states, q_target_batch, sample_weight=sample_weights
            )

            self.training_log.append({
                "training_round": self.training_round,
                "gradient_step": int(gradient_step),
                "loss": float(loss)
            })

        # update target network
        self.update_target_weights()
        self.training_round+= 1


class ConvolutionalDeepQAgent(DeepQAgent):

    def __init__(self, agent_id, n_actions, n_states, config):
        """
        A Q-learning agent with a convolutional network for Q-value approximation.

        :param agent_id:            The ID of the agent.
        :param n_actions:           The number of available actions.
        :param n_states:            The number of states in the environment.
        :param config:              A dictionary containing configuration parameters.
        """
        super().__init__(agent_id, n_actions, n_states, config)

        if self.board_encoding != self.ENCODING_TUTORIAL:
            raise ValueError("Only input in ENCODING_TUTORIAL (one-hot) supported in convolutional network")

        self.input_shape = (3, 3, 3)

    def state_to_input(self, state):
        # Convert to NHWC (batch size, height, width, number of channels)
        input = super().state_to_input(state)
        input = tf.reshape(input, (1,3,3,3))
        input = tf.transpose(input, [0,2,3,1])
        return input

    @tf.function(
        input_signature=[
            tf.TensorSpec([1, 3, 3, 3], tf.float32),  # NHWC conv input
            tf.TensorSpec([None],       tf.int32),    # legal action indices
        ]
    )
    def _graph_act(self, state, legal_idxs):
        # you can either re-implement the body, or call a shared helper
        q       = self.online_model(state, training=False)
        neg_inf = tf.constant(self.LARGE_NEGATIVE_NUMBER, tf.float32)
        mask = tf.scatter_nd(
          tf.expand_dims(legal_idxs, 1),
          tf.ones_like(legal_idxs, dtype=tf.float32),
          [self.n_actions]
        )
        mask = tf.reshape(mask, [1, -1])
        masked_q = mask * q + (1 - mask) * neg_inf
        return tf.argmax(masked_q, axis=1)[0]
