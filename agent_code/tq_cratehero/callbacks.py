import os
import pickle
import random

import numpy as np

from .agent_tabular_q import TabularQAgent

from .config import ACTIONS, N_ACTIONS, N_STATES
from .helpers import get_legal_actions, ascii_pictogram

config = {
    "n_episode"           : 50000,   # Number of training episodes
    "n_eval"              : 100,    # Number of evaluation episodes every eval_freq training episodes
    "eval_freq"           : 1000,
    "train_freq"          : 1,      # Train models every train_freq training episodes
    "discount"            : 0.95,   # Discount in all Q learning algorithms
    "learning_rate_decay" : 1,
    "exploration"         : 1.0,    # Initial exploration rate
    "exploration_decay"   : 1e-3,   # Decrease of exploration rate for every action
    "exploration_min"     : 0.2,
    "learning_rate"       : 1e-1,
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

    if not self.train and os.path.isfile("q-tables/q-table_190.pt.npz"):
        print("loadu")
        self.logger.info("Loading model from saved state.")
        self.agent.q = np.load("q-tables/q-table_190.pt.npz")["q"]


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """

    features = state_to_features(game_state)

    #Debugging
    #print(f"Feature bits: {features:015b}")      # 10-bit binary
    #print(ascii_pictogram(game_state))

    return ACTIONS[self.agent.act(features, actions=get_legal_actions(game_state=game_state))]

# features.py
import numpy as np
from collections import deque
from settings import BOMB_POWER
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]          # URDL

# ------------- helpers ------------------------------------------------------
def in_bounds(x, y, rows, cols):
    return 0 <= x < rows and 0 <= y < cols

def is_free(nx, ny, arena, bombs, others):
    return arena[nx, ny] == 0     \
       and all((nx, ny) != pos for pos,_ in bombs) \
       and all((nx, ny) != pos for *_n,pos in others)

def first_dir_bfs(start, goals, arena, bombs, others):
    """Return 0 (no goal) or 1-4 (URDL) for the nearest goal."""
    if not goals:
        return 0
    rows, cols = arena.shape
    Q = deque([(start, None)])
    seen = {start}
    while Q:
        (cx, cy), first = Q.popleft()
        if (cx, cy) in goals:
            return 0 if first is None else first + 1   # +1 so Up=1 …
        for d, (dx, dy) in enumerate(DIRS):
            nx, ny = cx + dx, cy + dy
            if not in_bounds(nx, ny, rows, cols):  continue
            if not is_free(nx, ny, arena, bombs, others): continue
            if (nx, ny) in seen:                   continue
            seen.add((nx, ny))
            Q.append(((nx, ny), d if first is None else first))
    return 0

# ---------------------------------------------------------------------------
def state_to_features(game_state):
    if game_state is None:
        return None

    arena      = game_state["field"]
    bombs      = game_state["bombs"]
    coins      = set(map(tuple, game_state["coins"]))
    others     = game_state["others"]
    expl_map   = game_state["explosion_map"]
    name, score, bombs_left, (x, y) = game_state["self"]
    rows, cols = arena.shape

    # ----- danger now
    danger_here = expl_map[x, y] > 0 or any((bx, by)==(x,y) and t==1 for (bx,by),t in bombs)

    # ----- safe-move mask (URDL)
    safe_mask = 0
    for d, (dx, dy) in enumerate(DIRS):
        nx, ny = x + dx, y + dy
        safe = in_bounds(nx, ny, rows, cols) \
            and is_free(nx, ny, arena, bombs, others) \
            and expl_map[nx, ny] == 0 \
            and all(not((bx, by)==(nx, ny) and t==1) for (bx,by),t in bombs)
        if safe:
            safe_mask |= 1 << d      # bit d

    # ----- nearest crate / enemy / coin (first-step code 0-4)
    crates  = {(cx, cy) for cx in range(rows) for cy in range(cols) if arena[cx, cy] == 1}
    enemies = {pos for *_n, pos in others}

    crate_dir = first_dir_bfs((x, y), crates,   arena, bombs, others)   # 0-4
    enemy_dir = first_dir_bfs((x, y), enemies, arena, bombs, others)   # 0-4
    coin_dir  = first_dir_bfs((x, y), coins,   arena, bombs, others)   # 0-4

    # ----- bomb available
    bomb_on_tile = any((bx, by)==(x, y) for (bx,by),_ in bombs)
    bomb_avail   = int(bombs_left > 0 and not bomb_on_tile)            # 0/1

    # ----- pack into 15-bit int
    state_id = (
        (bomb_avail       << 14) |
        (coin_dir  << 11) |
        (enemy_dir << 8)  |
        (crate_dir << 5)  |
        (safe_mask << 1)  |
        danger_here
    )
    return state_id
