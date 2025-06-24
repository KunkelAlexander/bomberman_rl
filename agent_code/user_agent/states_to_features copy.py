
# features.py
import numpy as np
from collections import deque
from settings import BOMB_POWER, BOMB_TIMER


# ---------------------------------------------------------------------------
# constants & small helpers (unchanged unless noted)
# ---------------------------------------------------------------------------
DIR_VECS    = [(0, -1), (1, 0), (0, 1), (-1, 0)]          # WURDL
DIRS        = ["UP", "RIGHT", "DOWN", "LEFT"]
ACTS        = DIRS + ["BOMB", "WAIT"]

DIR_BITS    = {i: a for i, a, in enumerate(DIRS)}
BITS_DIR    = {v: k for k, v in DIR_BITS.items()}  # inverse lookup
ACT_BITS    = {i: a for i, a, in enumerate(ACTS)}
BITS_ACT    = {v: k for k, v in ACT_BITS.items()}  # inverse lookup

# 2‑bit occupancy codes for the *immediate* neighbour tile
OCCS        = ["WALL", "COIN", "CRATE", "ENEMY"]
OCC_BITS    = {i: n for i, n in enumerate(OCCS)}
BITS_OCC    = {n: i for i, n in OCC_BITS.items()}

# 2‑bit encoding of object type of interest
OBJ_BITS = {
    "NONE"  : 0b00,
    "ENEMY" : 0b01,
    "CRATE" : 0b10,
    "COIN"  : 0b11,
}
BITS_OBJ = {v: k for k, v in OBJ_BITS.items()}

# ------------- helpers ------------------------------------------------------
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
        for dx, dy in DIR_VECS[1:]:
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
    if arena[nx, ny] != 0:                       # not walkable and therefore not safe
        return False
    if expl_map[nx, ny] > 0:                     # already exploding
        return False

    # --- and is there a full survival path for the existing bombs?
    return can_survive_current_bombs((nx, ny), arena, bombs, others, expl_map, blast_map)


# ---------------------------------------------------------------------------
# 17‑bit state encoding  ───────────────────────────────────
# layout (LSB→MSB):
#   0 – 7    neighbour occupancy (4 × 2 bits URDL)
#   8 – 13   safety bits for actions URDLWB (1 = safe action, 0 unsafe action)
#   14 – 15  direction (2 bits) of nearest object‑of‑interest
#   16 – 17  object type (2 bits) 00 none, 01 enemy, 10 crate, 11 coin
#
# total: 17 bits  (still fits into a Python int)
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


    # ---------------------------------------------------------------- per‑direction info
    neighbour_bits = 0   # 8 bits (2xURDL)
    safety_bits    = 0   # 6 bits (URDLBW)

    for d, (dx, dy) in enumerate(DIR_VECS[1:], start=1):   # d = 1..4 (URDL)
        nx, ny = x + dx, y + dy

        # ---- 3‑bit OCCUPANCY ----------------------------------------------
        if arena[nx, ny] == -1:
            occ = OCC_REV["WALL"]
        elif arena[nx, ny] == 1:
            occ = OCC_REV["CRATE"]
        elif any((nx, ny) == pos for *_n, pos in others):
            occ = OCC_REV["ENEMY"]
        elif (nx, ny) in coins:
            occ = OCC_REV["COIN"]
        else:
            occ = OCC_REV["EMPTY"]

        neighbour_bits |= (occ & 0b111) << (3 * (4 - d))  # 3 bits per dir

        # ---- SAFETY BIT  (1 = safe to stand, 0 = death / explosion / wall)
        if is_safe_tile(nx, ny, arena, bombs, expl_map, blast_map, others):
            safety_bits |= 1 << (d - 1)

    # ---------------------------------------------------------------- tile‑related bits
    # safe on current tile now?
    here_safe = is_safe_tile(x, y, arena, bombs, expl_map, blast_map, others)

    # would it still be safe after placing a bomb here?
    bombs_with_self     = list(bombs) + [((x, y), BOMB_TIMER)]
    blast_map_with_self = compute_blast_map(arena, bombs_with_self)
    here_safe_with_bomb = is_safe_tile(x, y, arena, bombs_with_self, expl_map,
                                       blast_map_with_self, others)

    bomb_on_tile = any((bx, by) == (x, y) for (bx, by), _ in bombs)
    bomb_avail   = int(bombs_left > 0 and not bomb_on_tile and here_safe_with_bomb)



    # nearest XYZ (dir=0 means none found)
    coin_dir,  coin_dist  = dir_and_dist_bfs((x, y), coins,
                                             arena, bombs, others)
    crate_dir, crate_dist = dir_and_dist_bfs_to_adjacent((x, y), crate_pos,
                                             arena, bombs, others)
    enemy_dir, enemy_dist = dir_and_dist_bfs((x, y), enemies,
                                             arena, bombs, others)

    #if coin_dir!= None:
    #    print(f"Direction to coin:  {DIR_NAME[coin_dir]}, {coin_dist}")
    #if crate_dir!= None:
    #    print(f"Direction to crate: {DIR_NAME[crate_dir]}, {crate_dist}")
    #if enemy_dir!= None:
    #    print(f"Direction to enemy: {DIR_NAME[enemy_dir]}, {enemy_dist}")
    # ---------- determine “object of interest” according to priority rules --
    obj_bits  = OBJ_BITS["NONE"]
    dir_bits  = DIR_BITS[0]

    # helper: convert dir (0‑4) -> two bits (0‑3)
    def dir_to_bits(d: int) -> int:
        return DIR_BITS[d]  # mapping already handles 0 → 00

    in_range_enemy = enemy_dist != None and enemy_dist <= 5

    if in_range_enemy:
        obj_bits = OBJ_BITS["ENEMY"]
        dir_bits = dir_to_bits(enemy_dir)
    elif coin_dist != None:
        obj_bits = OBJ_BITS["COIN"]
        dir_bits = dir_to_bits(coin_dir)
    elif crate_dist != None:
        obj_bits = OBJ_BITS["CRATE"]
        dir_bits = dir_to_bits(crate_dir)
    elif enemy_dist != None:  # farther enemies are the last resort
        obj_bits = OBJ_BITS["ENEMY"]
        dir_bits = dir_to_bits(enemy_dir)
    # else keep NONE / 00

    # ---------------------------------------------------------------- pack bits into int
    state_id = (
        (here_safe   << 22) |
        (bomb_avail  << 21) |
        (obj_bits    << 19) |
        (dir_bits    << 16) |
        (safety_bits << 12) |
        neighbour_bits
    )

    return state_id

# ---------------------------------------------------------------------------
# helper: human‑readable description of a 22‑bit state id
# ---------------------------------------------------------------------------

def describe_state(state_id: int) -> str:
    here_safe   = (state_id >> 22) & 1
    bomb_avail  = (state_id >> 21) & 1
    obj_bits    = (state_id >> 19) & 0b11
    dir_bits    = (state_id >> 16) & 0b111
    safety_bits = (state_id >> 12) & 0xF
    neighbour   = state_id & 0xFFF  # 12 lowest bits

    print("obj bits: ", obj_bits, " dir_bits: ", dir_bits)
    obj_name = BITS_OBJ[obj_bits]
    dir_name = DIR_NAME[BITS_DIR[dir_bits]]

    def bits_to_dirs(bits):
        return [name for d, name in enumerate(("UP", "RIGHT", "DOWN", "LEFT"), 1)
                if bits & (1 << (d - 1))]

    # decode neighbour occupancy
    neigh_occ = []
    for shift in (9, 6, 3, 0):
        code = (neighbour >> shift) & 0b111
        neigh_occ.append(OCC[code])

    return (
        f"{state_id:022b}\n"
        f"Nearest interest  : {obj_name} ({dir_name})\n"
        f"Safe actions      : {bits_to_dirs(safety_bits)}\n"
        f"Neighbour Up      : {neigh_occ[0]}\n"
        f"Neighbour Right   : {neigh_occ[1]}\n"
        f"Neighbour Down    : {neigh_occ[2]}\n"
        f"Neighbour Left    : {neigh_occ[3]}"
    )
