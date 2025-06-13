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

    if not self.train and os.path.isfile("q-tables/q-table_990.pt.npz"):
        print("loadu")
        self.logger.info("Loading model from saved state.")
        self.agent.q = np.load("q-tables/q-table_990.pt.npz")["q"]


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

    #print(f"Feature bits: {features:010b}")      # 10-bit binary
    #print(ascii_pictogram(game_state))

    return ACTIONS[self.agent.act(features, actions=get_legal_actions(game_state=game_state))]




# Cardinal helpers -----------------------------------------------------------
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]   # U, R, D, L

def in_bounds(x, y, rows, cols):
    return 0 <= x < rows and 0 <= y < cols

def free_of_bombs_agents(bx, by, bombs, others):
    """True iff no bomb or agent occupies (bx,by)."""
    return all((bx, by) != (x, y) for (x, y), _ in bombs) and \
           all((bx, by) != (x, y) for *_n, (x, y) in others)


def state_to_features(game_state: dict) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.


        state = {
            'round': self.round,
            'step': self.step,
            'field': np.array(self.arena),
            'self': agent.get_state(),
            'others': [other.get_state() for other in self.active_agents if other is not agent],
            'bombs': [bomb.get_state() for bomb in self.bombs],
            'coins': [coin.get_state() for coin in self.coins if coin.collectable],
            'user_input': self.user_input,
        }


        explosion_map = np.zeros(self.arena.shape)
        for exp in self.explosions:
            if exp.is_dangerous():
                for (x, y) in exp.blast_coords:
                    explosion_map[x, y] = max(explosion_map[x, y], exp.timer - 1)
        state['explosion_map'] = explosion_map

        agent.get_state: Provide information about this agent for the global game state
        return self.name, self.score, self.bombs_left, (self.x, self.y)

        coin.get_state(self):
        return self.x, self.y

        bomb.get_state(self):
        return (self.x, self.y), self.timer

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    if game_state is None:
        return None                        # called before game starts

    arena      = game_state["field"]       # -1 wall, 1 crate, 0 free
    bombs      = game_state["bombs"]       # [((x,y), timer), ...]
    coins      = game_state["coins"]       # [(x,y), ...]
    others     = game_state["others"]      # [(name, score, bombs_left, (x,y)), ...]
    explosions = game_state["explosion_map"]
    name, score, bombs_left, (x, y) = game_state["self"]
    rows, cols = arena.shape

    # -----------------------------------------------------------------------
    # 0. Danger at current tile
    danger_here = explosions[x, y] > 0

    # 1-4. Safe moves (free & not exploding next tick)
    safe_dirs = []
    for dx, dy in DIRS:
        nx, ny = x + dx, y + dy
        safe = in_bounds(nx, ny, rows, cols)           \
               and arena[nx, ny] == 0                  \
               and free_of_bombs_agents(nx, ny, bombs, others) \
               and explosions[nx, ny] == 0
        safe_dirs.append(safe)

    # 5-8. Coin direction flags
    USE_SHORTEST_PATH = True
    if not USE_SHORTEST_PATH:
        # Line of sight
        coin_dirs = [False, False, False, False]
        for cx, cy in coins:
            dx, dy = np.sign(cx - x), np.sign(cy - y)
            # same column (vertical) or same row (horizontal)
            if cx == x and dy != 0:
                step = (0, dy)
            elif cy == y and dx != 0:
                step = (dx, 0)
            else:
                continue
            # Walk from agent towards coin until blocked
            tx, ty = x + step[0], y + step[1]
            visible = True
            while (tx, ty) != (cx, cy):
                if arena[tx, ty] != 0:         # wall or crate blocks view
                    visible = False
                    break
                tx += step[0]; ty += step[1]
            if visible:
                coin_dirs[DIRS.index(step)] = True
    else:
        # -----------------------------------------------------------------------
        # 5-8. Coin direction flags (shortest-path first step)
        coin_dirs = [False, False, False, False]        # U, R, D, L

        # Breadth-first search limited to a modest radius (≤ 6 is enough)
        from collections import deque
        visited = set([(x, y)])
        queue   = deque([((x, y), None)])               # (pos, first_dir)

        while queue:
            (cx, cy), first_dir = queue.popleft()

            # Found a coin? record its originating direction
            if (cx, cy) in coins and first_dir is not None:
                coin_dirs[first_dir] = True
                break   # stop at nearest coin

            # expand neighbours
            for dir_idx, (dx, dy) in enumerate(DIRS):
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in visited:
                    continue
                if not in_bounds(nx, ny, rows, cols):
                    continue
                if arena[nx, ny] != 0:                  # wall or crate blocks
                    continue
                if not free_of_bombs_agents(nx, ny, bombs, others):
                    continue

                visited.add((nx, ny))
                queue.append(((nx, ny), dir_idx if first_dir is None else first_dir))


    # 9. Bomb available
    bomb_on_tile = any((bx, by) == (x, y) for (bx, by), _ in bombs)
    bomb_avail   = bombs_left > 0 and not bomb_on_tile

    # -----------------------------------------------------------------------
    # Encode to single integer key
    bits = [danger_here, *safe_dirs, *coin_dirs, bomb_avail]
    bits = np.array(bits, dtype=np.uint8)
    # little-endian binary-to-int: Σ b_i · 2ⁱ
    state_id = int((bits << np.arange(bits.size)).sum())

    #index (LSB→MSB):  9   8   7   6   5     4   3   2   1   0
    #feature:          B   CL  CD  CR  CU    SL  SD  SR  SU  D

    return state_id