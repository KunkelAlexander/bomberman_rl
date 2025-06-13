import numpy as np
from .config import ACTION_STRING_TO_ID, N_ACTIONS   # (‘UP’, ‘RIGHT’, ‘DOWN’, ‘LEFT’, ‘WAIT’, ‘BOMB’)

DIR2VEC = {
    'UP'   : ( 0, -1),
    'RIGHT': ( 1,  0),
    'DOWN' : ( 0,  1),
    'LEFT' : (-1,  0),
}

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
    for act, (dx, dy) in DIR2VEC.items():
        if tile_is_free(x + dx, y + dy):
            legal.append(ACTION_STRING_TO_ID[act])

    # WAIT is always legal
    legal.append(ACTION_STRING_TO_ID['WAIT'])

    # BOMB: at least one bomb left *and* no bomb already on current tile
    if bombs_left > 0 and all((bx, by) != (x, y) for (bx, by), _ in bombs):
        legal.append(ACTION_STRING_TO_ID['BOMB'])

    return np.array(legal, dtype=int)

###############################################################################
# debug_vis.py
###############################################################################
def ascii_pictogram(game_state: dict) -> str:
    """
    Build a human-readable ASCII map for quick visual debugging.
    Legend
        # : indestructible wall          (-1 in arena)
        + : crate                        ( 1 in arena)
          : free tile                    ( 0 in arena)
        A : *your* agent
        O : other agent
        C : coin
        B : bomb  (timer shown as digit 1-4)
        x : tile that will explode next tick   (explosion_map[x,y] > 0)
    """
    if game_state is None:
        return "<no board yet>"

    arena      = game_state["field"]
    bombs      = {pos: t for pos, t in game_state["bombs"]}
    coins      = {tuple(c) for c in game_state["coins"]}
    explosions = game_state["explosion_map"]
    me_name, *_ignore, (ax, ay) = game_state["self"]
    others     = {tuple(o[-1]) for o in game_state["others"]}

    rows, cols = arena.shape
    pict = []

    for y in range(cols):              # NOTE: arena is (x,y); y=rows is vertical axis
        row_chars = []
        for x in range(rows):
            ch = " "
            if arena[x, y] == -1:       ch = "#"
            elif arena[x, y] ==  1:     ch = "+"
            if explosions[x, y] > 0:    ch = "x"
            if (x, y) in coins:         ch = "C"
            if (x, y) in others:        ch = "O"
            if (x, y) == (ax, ay):      ch = "A"
            if (x, y) in bombs:
                timer = min(bombs[(x, y)], 4)   # clip to 4 so it fits in one char
                ch = str(timer) if ch == " " else ch
            row_chars.append(ch)
        pict.append("".join(row_chars))
    return "\n".join(pict)

###############################################################################
# debug_vis.py
###############################################################################
DIRS = ["Up", "Right", "Down", "Left"]          # index 0-3

def _decode_dir(code: int) -> str:
    """0-4 → human-readable direction"""
    return "None" if code == 0 else DIRS[code - 1]
DIRS = ["Up", "Right", "Down", "Left"]          # 0-3 → URDL

def _decode_dir(code: int) -> str:
    """0-4 → None / Up / Right / …"""
    return "None" if code == 0 else DIRS[code - 1]

def _decode_safe(code: int) -> str:
    """0-5 → human text for the SAFE_DIR field."""
    table = {
        0: "!!! no safe option (tile lethal)",
        1: "WAIT is safe",
        2: "Up",
        3: "Right",
        4: "Down",
        5: "Left",
    }
    return table.get(code, "reserved")

# ┌ bit 12 ────────────── bit 0 ┐
# │b│ coin │ enemy │ crate │SAFE│
# │ │ 11-9 │ 8-6   │ 5-3   │2-0 │
def describe_state(state_id: int) -> str:
    safe_dir   =  state_id        & 0b111          # 0-5
    crate_dir  = (state_id >> 3)  & 0b111          # 0-4
    enemy_dir  = (state_id >> 6)  & 0b111
    coin_dir   = (state_id >> 9)  & 0b111
    bomb_avail =  bool(state_id >> 12)

    lines = [
        f"Raw bits         : {state_id:013b}",
        f"Bomb available   : {'YES' if bomb_avail else 'no'}",
        f"SAFE choice      : {_decode_safe(safe_dir)}",
        f"Nearest crate    : {_decode_dir(crate_dir)}",
        f"Nearest enemy    : {_decode_dir(enemy_dir)}",
        f"Nearest coin     : {_decode_dir(coin_dir)}",
    ]
    return "\n".join(lines)




# features.py
import numpy as np
from collections import deque
from settings import BOMB_POWER
DIR_VECS = [(0, -1), (1, 0), (0, 1), (-1, 0)]          # URDL

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
        for d, (dx, dy) in enumerate(DIR_VECS):
            nx, ny = cx + dx, cy + dy
            if not in_bounds(nx, ny, rows, cols):  continue
            if not is_free(nx, ny, arena, bombs, others): continue
            if (nx, ny) in seen:                   continue
            seen.add((nx, ny))
            Q.append(((nx, ny), d if first is None else first))
    return 0


import numpy as np
from settings import BOMB_POWER        # 3 by default

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
# ---------------------------------------------------------------------------
DIR_VECS  = [(0, -1), (1, 0), (0, 1), (-1, 0)]   # Up, Right, Down, Left
#  2-5 are the SAFE_DIR codes for URDL
DIR_CODE  = {v: i + 2 for i, v in enumerate(DIR_VECS)}

def best_safe_dir(x, y, arena, bombs, expl_map, blast_map):
    """
    Return SAFE_DIR code:
      0 = no survivable action (do the least-bad move later)
      1 = WAIT is safe
      2..5 = Up / Right / Down / Left  (best step)
    """
    rows, cols = arena.shape
    t_here = blast_map[x, y]

    # ------------------------------------------------ 1️⃣  WAIT?
    # Wait only if the tile will stay intact ≥4 ticks (or never blows)
    if expl_map[x, y] == 0 and (t_here >= 4 or t_here == 99):
        return 1          # code for WAIT safe

    # ------------------------------------------------ 2️⃣  Evaluate moves
    best_code  = 0        # default: no safe move
    best_timer = -1

    for code, (dx, dy) in enumerate(DIR_VECS, start=2):   # Up=2 …
        nx, ny = x + dx, y + dy
        if not (0 <= nx < rows and 0 <= ny < cols):
            continue
        if arena[nx, ny] != 0 or expl_map[nx, ny] > 0:
            continue
        if any((nx, ny) == pos for pos, _ in bombs):
            continue

        t = blast_map[nx, ny]          # ticks before blast on that tile
        if t == 0:                     # detonates next tick → skip
            continue

        # choose the neighbour with the *largest* timer
        # ties broken by DIR_VECS order (Up > Right > Down > Left)
        if t > best_timer:
            best_timer, best_code = t, code

    return best_code    # 0 if nothing beats certain death



"""
| Field                      | Values               | Bits        | Comment             |
| -------------------------- | -------------------- | ----------- | ------------------- |
| **DANGER now**             | 0/1                  | 1           | as before           |
| **SAFE-move mask** U R D L | 16                   | 4           | 1 = walkable & safe |
| **DIR to nearest *crate*** | 0 = none, 1-4 = URDL | **3**       | BFS first-step      |
| **DIR to nearest *enemy*** | 0/1-4                | **3**       | BFS first-step      |
| **DIR to nearest *coin***  | 0/1-4                | **3**       | BFS first-step      |
| **BOMB available**         | 0/1                  | 1           | as before           |
| **Total**                  | –                    | **15 bits** | 2¹⁵ = 32 768 states |

bit 14  13‒11  10‒8   7‒5    4‒1    0
      ─┬─────┬─────┬──────┬──────┬────
       │ coin │enemy│crate │ safe │DANGER
       │ dir  │ dir │ dir  │mask  │
BOMB-avail is now the **top** bit (14)

dir code: 0 = none / 1 = Up / 2 = Right / 3 = Down / 4 = Left

32 768
"""
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

    blast_map  = compute_blast_map(arena, bombs)

    # ---------- SAFE_DIR (0-5)
    safe_dir = best_safe_dir(x, y, arena, bombs, expl_map, blast_map)

    # ---------- nearest crate / enemy / coin  (0-4)
    crates  = {(cx, cy) for cx in range(rows) for cy in range(cols)
               if arena[cx, cy] == 1}
    enemies = {pos for *_n, pos in others}

    crate_dir = first_dir_bfs((x, y), crates,  arena, bombs, others)   # 0-4
    enemy_dir = first_dir_bfs((x, y), enemies, arena, bombs, others)   # 0-4
    coin_dir  = first_dir_bfs((x, y), coins,   arena, bombs, others)   # 0-4

    # ---------- bomb availability bit
    bomb_on_tile = any((bx, by) == (x, y) for (bx, by), _ in bombs)
    bomb_avail   = int(bombs_left > 0 and not bomb_on_tile)            # 0/1

    # ---------- pack into 13-bit int
    state_id = (
        (bomb_avail << 12) |
        (coin_dir   << 9)  |
        (enemy_dir  << 6)  |
        (crate_dir  << 3)  |
        safe_dir
    )
    return state_id
