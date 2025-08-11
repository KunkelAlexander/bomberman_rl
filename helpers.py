
# features.py
import numpy as np
from collections import deque
from enum import IntEnum

import os
import sys

from settings import BOMB_POWER, BOMB_TIMER

from typing import List
import events as e

# ---------------------------------------------------------------------------
# constants & small helpers (unchanged unless noted)
# ---------------------------------------------------------------------------


class TransitionFields(IntEnum):
    ITERATION          = 0
    STATE              = 1
    LEGAL_ACTIONS      = 2
    ACTION             = 3
    REWARD             = 4
    DONE               = 5
    NEXT_STATE         = 6
    NEXT_LEGAL_ACTIONS = 7

DIR_VECS    = [(0, -1), (1, 0), (0, 1), (-1, 0)]          # WURDL
DIRS        = ["UP", "RIGHT", "DOWN", "LEFT"]
ACTS        = DIRS + ["BOMB", "WAIT"]
OBJS        = ["NONE", "ENEMY", "CRATE", "COIN"]
OCCS        = ["EMPTY", "WALL", "COIN", "CRATE", "ENEMY", "BOMB", "EXPLOSION"]

DIR_BITS    = {a: i for i, a in enumerate(DIRS)}
ACT_BITS    = {a: i for i, a in enumerate(ACTS)}
OBJ_BITS    = {a: i for i, a in enumerate(OBJS)}
OCC_BITS    = {a: i for i, a in enumerate(OCCS)}

N_ACTIONS   = len(ACTS)
N_STATES    = 2**22

# ------------- helpers ------------------------------------------------------
def print_game_ascii(game_state):
    """
    Prints an ASCII representation of the full game state for debugging purposes.
    Legend:
      # = Wall
      % = Crate
      . = Free tile
      x = Explosion
      b = Bomb (with timer)
      C = Coin
      P = Player (yourself)
      E = Enemy
    """
    if game_state is None:
        print("No game state available.")
        return

    field = game_state['field'].copy()
    expl_map = game_state['explosion_map']
    bombs = dict(game_state['bombs'])  # {(x, y): timer}
    coins = set(game_state['coins'])
    others = {pos for *_n, pos in game_state['others']}
    self_pos = game_state['self'][-1]

    rows, cols = field.shape
    display = [[' ' for _ in range(cols)] for _ in range(rows)]

    for x in range(rows):
        for y in range(cols):
            if field[x, y] == -1:
                display[x][y] = '#'
            elif field[x, y] == 1:
                display[x][y] = '%'
            elif expl_map[x, y] > 0:
                display[x][y] = 'x'
            else:
                display[x][y] = '.'

    for (bx, by), t in game_state['bombs']:
        display[bx][by] = str(t)

    for cx, cy in coins:
        display[cx][cy] = 'C'

    for ox, oy in others:
        display[ox][oy] = 'E'

    px, py = self_pos
    display[px][py] = 'P'

    print("\nASCII game state:")
    for y in range(cols):
        line = ""
        for x in range(rows):
            line += display[x][y]
        print(line)


def in_bounds(x, y, rows, cols):
    return 0 <= x < rows and 0 <= y < cols

def is_free(nx, ny, arena, bombs, others):
    return arena[nx, ny] == 0     \
       and all((nx, ny) != pos for pos,_ in bombs) \
       and all((nx, ny) != pos for *_n,pos in others)

# ---------- new: BFS variant that also gives the distance ------------------
def dir_and_dist_bfs(start, goals, arena, bombs, others):
    """
    Return (dir, dist) where
        dir  = 0-4 (WURDL) for the *nearest* goal (None if no goal)
        dist = #steps to that goal   (None if no goal)
    """
    if not goals:
        return None, None
    rows, cols = arena.shape
    Q = deque([(start, None, 0)])
    seen = {start}
    while Q:
        (cx, cy), first, dist = Q.popleft()
        if (cx, cy) in goals:
            return (0 if first is None else first), dist
        for d, (dx, dy) in enumerate(DIR_VECS):
            nx, ny = cx + dx, cy + dy
            if not in_bounds(nx, ny, rows, cols):  continue
            if not arena[nx, ny] == 0:             continue
            if (nx, ny) in seen:                   continue
            seen.add((nx, ny))
            Q.append(((nx, ny), d if first is None else first, dist + 1))
    return None, None

def dir_and_dist_bfs_to_adjacent(start, goals, arena, bombs, others):
    """Same as above, but goals = *tiles adjacent to any crate*."""
    if not goals:
        return None, None
    rows, cols = arena.shape
    # collect all passable neighbours of every crate
    adj = set()
    for gx, gy in goals:
        for dx, dy in DIR_VECS:
            nx, ny = gx + dx, gy + dy
            if in_bounds(nx, ny, rows, cols) and \
               arena[nx, ny] == 0:
                adj.add((nx, ny))
    return dir_and_dist_bfs(start, adj, arena, bombs, others)

def compute_blast_map(arena, bombs):
    """
    Returns an int array M where:
        M[x,y] = smallest timer of any bomb that will blast (x,y)
                 or 99 if no bomb reaches it.
    """
    rows, cols = arena.shape
    INF = 99
    M   = np.full((rows, cols), INF, dtype=np.int8)

    for (bx, by), t in bombs:
        for dx, dy in [(0,0), (1,0), (-1,0), (0,1), (0,-1)]:
            for k in range(0 if dx==dy==0 else 1, BOMB_POWER+1):
                x, y = bx + dx*k, by + dy*k
                if not (0 <= x < rows and 0 <= y < cols):
                    break
                if arena[x, y] == -1:        # solid wall blocks blast and ray
                    break
                M[x, y] = min(M[x, y], t)
                if arena[x, y] == 1:         # crate stops blast further on
                    break
    return M          # 0/1/2/3/4/… / 99



def can_survive_current_bombs(start_xy, arena, bombs, others, expl_map, blast_map) -> bool:
    """
    Return True  → there exists a sequence of moves/WAITs that avoids
                   every blast until all current bombs are finished.
           False → every possible action sequence ends in an explosion.
    """
    if not bombs:                         # trivial: no bombs at all
        return True

    # --- how long do we need to survive?
    last_timer = max(t for _pos, t in bombs)          # 1…4
    horizon    = last_timer                           # ticks to simulate
    rows, cols = arena.shape
    DIR_WAIT   = DIR_VECS + [(0, 0)]                  # 4 moves + WAIT

    Q     = deque([(start_xy[0], start_xy[1], 0)])    # (x, y, t)
    seen  = {(start_xy, 0)}

    while Q:
        x, y, t = Q.popleft()

        # blown up already?
        if blast_map[x, y] <= t or expl_map[x, y] > 0 > t:
            continue

        # reached or passed horizon → survived!
        if t >= horizon:
            return True

        # expand to next tick
        for dx, dy in DIR_WAIT:
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny, rows, cols):
                continue
            if arena[nx, ny] != 0:
                continue
            if any((nx, ny) == pos for pos, _ in bombs):
                continue
            if any((nx, ny) == pos for *_n, pos in others):
                continue

            key = (nx, ny, t + 1)
            if key not in seen:
                seen.add(key)
                Q.append((nx, ny, t + 1))

    # queue exhausted → every path died
    return False

def is_safe_tile(nx: int, ny: int,
                 arena: np.ndarray,
                 bombs, expl_map, blast_map, others) -> bool:
    """
    True  → stepping onto (nx,ny) still leaves at least ONE survivable action
            (including WAIT) according to best_safe_dir from that tile.
    False → certain death.
    """
    if arena[nx, ny] != 0:                       # not walkable and therefore safe
        return True
    if expl_map[nx, ny] > 0:                     # already exploding
        return False

    # --- and is there a full survival path for the existing bombs?
    return can_survive_current_bombs((nx, ny), arena, bombs, others, expl_map, blast_map)


# ---------------------------------------------------------------------------
# NEW   ──  compact 22‑bit state encoding  ───────────────────────────────────
# layout (LSB→MSB):
#   0 – 11   neighbour occupancy (4 × 3 bits URDL)
#   12 – 13  direction (2 bits) of nearest object‑of‑interest 00 up, 01 right, 10 down, 11 left
#   14 – 15  object type (2 bits) 00 none, 01 enemy, 10 crate, 11 coin
#   16 – 21  safety bits for URDLBW (1 = safe action, 0 = impassable or deadly)
#
# total: 22 bits
# ---------------------------------------------------------------------------


def state_to_features(game_state: dict | None) -> int | None:
    if game_state is None:
        return None

    arena      = game_state["field"]
    bombs      = game_state["bombs"]
    coins      = set(map(tuple, game_state["coins"]))
    others     = game_state["others"]
    expl_map   = game_state["explosion_map"]
    name, score, bombs_left, (x, y) = game_state["self"]
    rows, cols = arena.shape
    # ---------------------------------------------------------------- helpers
    blast_map  = compute_blast_map(arena, bombs)
    crate_pos  = {(cx, cy) for cx in range(rows) for cy in range(cols)
                  if arena[cx, cy] == 1}
    enemies    = {pos for *_n, pos in others}

    # nearest XYZ (dir=0 means none found)
    coin_dir,  coin_dist  = dir_and_dist_bfs((x, y), coins,
                                             arena, bombs, others)
    crate_dir, crate_dist = dir_and_dist_bfs_to_adjacent((x, y), crate_pos,
                                             arena, bombs, others)
    enemy_dir, enemy_dist = dir_and_dist_bfs((x, y), enemies,
                                             arena, bombs, others)

    # ---------- determine “object of interest” according to priority rules --
    obj_bits  = OBJ_BITS["NONE"]
    dir_bits  = DIR_BITS["UP"]

    if enemy_dist != None:
        obj_bits = OBJ_BITS["ENEMY"]
        dir_bits = enemy_dir
    elif coin_dist != None:
        obj_bits = OBJ_BITS["COIN"]
        dir_bits = coin_dir
    elif crate_dist != None:
        obj_bits = OBJ_BITS["CRATE"]
        dir_bits = crate_dir
    # else keep NONE / 00

    # ---------------------------------------------------------------- per‑direction info
    neighbour_bits = 0   # 12 bits
    safety_bits    = 0   # 6  bits (URDLWB)

    for d, (dx, dy) in enumerate(DIR_VECS, start=1):   # d = 1..4 (URDL)
        nx, ny = x + dx, y + dy

        # ---- 3‑bit OCCUPANCY ----------------------------------------------
        if arena[nx, ny] == -1:
            occ = OCC_BITS["WALL"]
        elif arena[nx, ny] == 1:
            occ = OCC_BITS["CRATE"]
        elif any((nx, ny) == pos for *_n, pos in others):
            occ = OCC_BITS["ENEMY"]
        elif (nx, ny) in coins:
            occ = OCC_BITS["COIN"]
        else:
            occ = OCC_BITS["EMPTY"]

        neighbour_bits |= (occ & 0b111) << (3 * (4 - d))  # 3 bits per dir

        # ---- SAFETY BIT  (1 = safe to stand, 0 = death / explosion / wall)
        if is_safe_tile(nx, ny, arena, bombs, expl_map, blast_map, others):
            safety_bits |= 1 << (d - 1)

    # ---------------------------------------------------------------- tile‑related bits

    # would it still be safe after placing a bomb here?
    bombs_with_self     = list(bombs) + [((x, y), BOMB_TIMER)]
    blast_map_with_self = compute_blast_map(arena, bombs_with_self)
    here_safe_with_bomb = is_safe_tile(x, y, arena, bombs_with_self, expl_map,
                                       blast_map_with_self, others)

    # check if agent is standing on a bomb
    bomb_on_tile = any((bx, by) == (x, y) for (bx, by), _ in bombs)
    # is (placing a bomb allowed & there is no bomb on the current tile & placing a bomb would not kill us)?
    bomb_bit    = int(bombs_left > 0 and not bomb_on_tile and here_safe_with_bomb)

    # set 5th bit
    if bomb_bit:
        safety_bits |= 1 << 4

    # is waiting safe?
    wait_bit = is_safe_tile(x, y, arena, bombs, expl_map, blast_map, others)

    # set 6th bit
    if wait_bit:
        safety_bits |= 1 << 5

    # ---------------------------------------------------------------- pack bits into int
    state_id = (
        (safety_bits  << 16) |
        (obj_bits     << 14) |
        (dir_bits     << 12) |
        neighbour_bits
    )

    return state_id

# ---------------------------------------------------------------------------
# helper: human‑readable description of a 22‑bit state id
# ---------------------------------------------------------------------------

def describe_state(state_id: int) -> str:
    safety_bits     = (state_id >> 16) & 0b111111 # 6 safety bits for ["UP", "RIGHT", "DOWN", "LEFT", "BOMB", "WAIT"]
    obj_bits        = (state_id >> 14) & 0b11     # 2 bits for ["NONE", "ENEMY", "CRATE", "COIN"]
    dir_bits        = (state_id >> 12) & 0b11     # 2 bits for ["UP", "RIGHT", "DOWN", "LEFT"]
    neighbour_bits  = (state_id >>  0) & 0xFFF    # 12 bits for four directions ["UP", "RIGHT", "DOWN", "LEFT"] with the 3-bit states ["EMPTY", "WALL", "COIN", "CRATE", "ENEMY", "BOMB", "EXPLOSION"]
    obj_name        = OBJS[obj_bits]
    dir_name        = DIRS[dir_bits]

    safe_actions = [name for d, name in enumerate(ACTS) if safety_bits & (1 << d)]

    # decode neighbour occupancy
    neigh_occ = []
    for shift in (9, 6, 3, 0):
        code = (neighbour_bits >> shift) & 0b111  # extract 3 bits at a time
        neigh_occ.append(OCCS[code])

    return (
        f"{state_id:022b}\n"
        f"Nearest interest  : {obj_name} ({dir_name})\n"
        f"Safe actions      : {safe_actions}\n"
        f"Neighbour Up      : {neigh_occ[0]}\n"
        f"Neighbour Right   : {neigh_occ[1]}\n"
        f"Neighbour Down    : {neigh_occ[2]}\n"
        f"Neighbour Left    : {neigh_occ[3]}"
    )



def reward_from_events(events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED:  0.2,
        e.KILLED_OPPONENT: 1.0,
        e.CRATE_DESTROYED: 0.1,
        e.KILLED_SELF:    -1.0,
        e.SURVIVED_ROUND:  1.0,
        e.GOT_KILLED:     -0.1,
        e.WAITED:         -0.02,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    return reward_sum


def get_legal_actions(game_state) -> np.ndarray:
    """
    Return the list of ACTION IDs that are *physically legal* in the current Bomberman state.
    Legal = the move keeps you on the board, lands on a free tile (0 in `arena`)
            with no bomb and no other agent occupying it.
            'WAIT' is always legal.
            'BOMB' is legal iff you still have bombs_left and no bomb already on your tile.
    Explosion/danger is NOT checked here – leave that to the policy.
    """
    # If we have no game state (first call or game over)--everything is allowed.
    if game_state is None:
        return np.arange(N_ACTIONS)

    arena      = game_state["field"]          # 2-D int8 array, -1: wall, 1: crate, 0: free
    bombs      = game_state["bombs"]          # [((x, y), timer), …]
    others     = game_state["others"]         # [(name, score, bombs_left, (x, y)), …]
    name, score, bombs_left, (x, y) = game_state["self"]
    rows, cols = arena.shape

    # ----- helpers -----------------------------------------------------------
    def in_bounds(cx, cy):
        return 0 <= cx < rows and 0 <= cy < cols

    def tile_is_free(cx, cy):
        """Free = no wall/crate, no bomb, no other agent."""
        if not in_bounds(cx, cy):
            return False
        if arena[cx, cy] != 0:                # wall or crate
            return False
        for (bx, by), _ in bombs:
            if bx == cx and by == cy:
                return False
        for *_ignore, (ox, oy) in others:
            if ox == cx and oy == cy:
                return False
        return True
    # -------------------------------------------------------------------------

    legal = []

    # Movement actions
    for act, (dx, dy) in zip(DIRS, DIR_VECS):
        if tile_is_free(x + dx, y + dy):
            legal.append(ACT_BITS[act])

    # WAIT is always legal
    legal.append(ACT_BITS['WAIT'])

    # BOMB: at least one bomb left *and* no bomb already on current tile
    if bombs_left > 0 and all((bx, by) != (x, y) for (bx, by), _ in bombs):
        legal.append(ACT_BITS['BOMB'])

    return np.array(legal, dtype=int)

