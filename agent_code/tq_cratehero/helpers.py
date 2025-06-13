import numpy as np
from .config import ACTION_STRING_TO_ID, N_ACTIONS   # (‘UP’, ‘RIGHT’, ‘DOWN’, ‘LEFT’, ‘WAIT’, ‘BOMB’)

DIRS = {
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
    for act, (dx, dy) in DIRS.items():
        if tile_is_free(x + dx, y + dy):
            legal.append(ACTION_STRING_TO_ID[act])

    # WAIT is always legal
    legal.append(ACTION_STRING_TO_ID['WAIT'])

    # BOMB: at least one bomb left *and* no bomb already on current tile
    if bombs_left > 0 and all((bx, by) != (x, y) for (bx, by), _ in bombs):
        pass# Ignore bombs for nowlegal.append(ACTION_STRING_TO_ID['BOMB'])

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
