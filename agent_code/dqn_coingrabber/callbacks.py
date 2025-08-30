import os
import pickle
import random

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from q_deep_agent import DeepQAgent, ConvolutionalDeepQAgent

from q_helpers import get_legal_actions, ACTS, N_ACTIONS, N_STATES, state_to_features, describe_state

base_config = {
    "n_eval"              : 100,    # Number of evaluation episodes every eval_freq training episodes
    "eval_freq"           : 100,
    "train_freq"          : 1,      # Train models every train_freq training episodes
    "grad_steps"          : 2,      # Number of gradient updates per training step
    "discount"            : 0.8,    # Discount in all Q learning algorithms
    "learning_rate_decay" : 1,
    "exploration"         : 1.0,    # Initial exploration rate
    "exploration_decay"   : 1e-2,   # Decrease of exploration rate for every action
    "exploration_min"     : 0.1,
    "learning_rate"       : 1e-3,
    "debug"               : False,  # Print loss and evaluation information during training
    "plot_debug"          : False,  # Plot game outcomes
    "batch_size"          : 128,    # Batch size for DQN algorithm
    "board_encoding"      : "encoding_binary",
    "replay_buffer_size"  : 10000,  # Replay buffer for DQN algorithm
    "replay_buffer_min"   : 1000,   # minimum size before we start training
    "target_update_tau"   : 0.1,    # Weight for update in dual DQN architecture target = (1 - tau) * target + tau * online
    "target_update_freq"  : 10,     # Update target network every n episodes
    "target_update_mode"  : "hard", # "hard": update every target_update freq or "soft": update using Polyakov rule with target_update_tau
}




def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # Q-network factory
    def build_simple_dqn_model(input_shape, num_actions, num_hidden_layers=1, hidden_layer_size=128):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_hidden_layers):
            x = layers.Dense(hidden_layer_size, activation='relu')(x)
        outputs = layers.Dense(num_actions, activation='linear')(x)
        return models.Model(inputs=inputs, outputs=outputs)

    self.agent = DeepQAgent(
            agent_id=0,
            n_actions=N_ACTIONS,
            n_states=N_STATES,
            config=base_config,
    )
    hidden_layer_size = 128

    net = build_simple_dqn_model(self.agent.input_shape, self.agent.n_actions, hidden_layer_size=hidden_layer_size)
    net.compile(optimizer=tf.keras.optimizers.Adam(base_config["learning_rate"]), loss="mse")
    self.agent.online_model = net
    self.agent.target_model = net

    if not self.train:
        # Load everything back
        self.agent.load("./snapshots", base_name="experiment_01")






def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    state = state_to_features(game_state)
    return ACTS[self.agent.act(state, actions=get_legal_actions(game_state=game_state))]

