import os
import pickle
import random

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

from q_deep_agent import DeepQAgent

from q_helpers import get_legal_actions, ACTS, N_ACTIONS, N_STATES, state_to_tabular_features, describe_tabular_state

base_config = {
    "n_episode"           : 2000,
    "n_eval"              : 100,    # Number of evaluation episodes every eval_freq training episodes
    "eval_freq"           : 100,
    "train_freq"          : 1,      # Train models every train_freq training episodes
    "grad_steps"          : 2,      # Number of gradient updates per training step
    "discount"            : 0.8,    # Discount in all Q learning algorithms
    "learning_rate_decay" : 1,
    "exploration"         : 1.0,    # Initial exploration rate
    "exploration_decay"   : 1e-3,   # Decrease of exploration rate for every action
    "exploration_min"     : 0.1,
    "learning_rate"       : 1e-3,
    "debug"               : False,  # Print loss and evaluation information during training
    "plot_debug"          : False,  # Plot game outcomes
    "batch_size"          : 128,    # Batch size for DQN algorithm
    "board_encoding"      : "encoding_cnn",
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


    def build_cnn_dqn_model(input_shape, num_actions):
        """
        CNN DQN for Bomberman-like 9x9 grid inputs.

        input_shape: (rows, cols, channels), e.g. (9, 9, 11)
        num_actions: number of discrete actions
        """
        inputs = layers.Input(shape=input_shape)  # (9, 9, channels)

        # Conv stack
        x = layers.Conv2D(16, kernel_size=3, padding="same")(inputs)  # fewer filters
        x = layers.ReLU()(x)

        x = layers.Conv2D(32, kernel_size=3, padding="same")(x)
        x = layers.ReLU()(x)

        # Flatten and dense head
        x = layers.Flatten()(x)                     # 9*9*32 = 2592 units
        x = layers.Dense(64, activation="relu")(x)  # small dense layer

        # Output Q-values for each action
        outputs = layers.Dense(num_actions, activation="linear")(x)

        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    self.agent = DeepQAgent(
            agent_id=0,
            n_actions=N_ACTIONS,
            n_states=N_STATES,
            config=base_config,
    )

    self.agent.online_model = build_cnn_dqn_model(self.agent.input_shape, self.agent.n_actions)
    self.agent.online_model.compile(optimizer=tf.keras.optimizers.Adam(
        base_config["learning_rate"]),
        loss=tf.keras.losses.Huber()
    )
    self.agent.target_model = build_cnn_dqn_model(self.agent.input_shape, self.agent.n_actions)

    if not self.train:
        # Load everything back
        self.agent.load("./snapshots", base_name="default")






def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    return ACTS[self.agent.act(game_state, actions=get_legal_actions(game_state=game_state))]

