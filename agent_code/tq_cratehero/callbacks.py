import os
import pickle
import random

import numpy as np

from .agent_tabular_q import TabularQAgent

from .config import ACTIONS, N_ACTIONS, N_STATES
from .helpers import get_legal_actions, ascii_pictogram
from .states_to_features import state_to_features, describe_state

config = {
    "n_episode"           : 50000,   # Number of training episodes
    "n_eval"              : 100,    # Number of evaluation episodes every eval_freq training episodes
    "eval_freq"           : 1000,
    "train_freq"          : 1,      # Train models every train_freq training episodes
    "discount"            : 0.9,    # Discount in all Q learning algorithms
    "learning_rate_decay" : 1,
    "exploration"         : 1.0,    # Initial exploration rate
    "exploration_decay"   : 1e-3,   # Decrease of exploration rate for every action
    "exploration_min"     : 0.0,
    "learning_rate"       : 0.05,
    "learning_rate_decay" : 1,
    "randomise_order"     : False,  # Randomise starting order of agents for every game
    "only_legal_actions"  : True,   # Have agents only take legal actions
    "debug"               : False,  # Print loss and evaluation information during training
    "initial_q"           : 0.6,    # Initial Q value for tabular Q learning
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

    self.agent = TabularQAgent(
            agent_id=0,
            n_actions=N_ACTIONS,
            n_states=N_STATES,
            config=config,
    )


    if not self.train and os.path.isfile("q-tables/q-table_1750.pt.npz"):
        print("loadu")
        self.logger.info("Loading model from saved state.")
        self.agent.q = np.load("q-tables/q-table_1750.pt.npz")["q"]


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    features = state_to_features(game_state)

    if not self.train:

        features = state_to_features(game_state)
        print(describe_state(features))

    return ACTIONS[self.agent.act(features, actions=get_legal_actions(game_state=game_state))]

