# Q‑Learning with **BombeRLe**

This repository implements both **tabular Q‑learning** and a **Deep Q‑Network (DQN)** for the classic game *Bomberman* using the BombeRLe environment.

![Gameplay](figures/1_gameplay.gif)

---

## Table of contents

* [Project structure](#project-structure)
* [Agents](#agents)

  * [tq\_demonstrator (data recorder)](#tq_demonstrator-data-recorder)
  * [Exploration logic used for data collection](#exploration-logic-used-for-data-collection)
* [Usage](#usage)

  * [Coingrabber](#coingrabber)
  * [Crate‑hero](#crate-hero)
  * [Allstar](#allstar)
* [Training / code layout](#training--code-layout)
* [Tips & notes](#tips--notes)
* [License](#license)
* [Citation](#citation)

---

## Project structure

All **new agents** live under the `agent_code/` directory. A typical layout looks like this (truncated):

```text
.
├── agent_code/
│   ├── peaceful_agent/
│   ├── rule_based_agent/
│   ├── tq_demonstrator/        # NEW: rule_based-derived agent that records transitions
│   ├── tq_coingrabber/
│   ├── tq_cratehero/
│   └── tq_allstar/
├── q_agent_parent.py
├── q_build_training_episodes.py
├── q_deep_agent.py
├── q_optuna.py
├── q_pretrain_tabular.py
├── q_prioritised_experience_replay.py
├── q_tabular_agent.py          # Core tabular Q-learning implementation
├── q_train_tabular.py
└── figures/
```

> **Heads‑up:** paths below assume you run commands from the repo root.

---

## Agents

* **tq\_coingrabber** – agent trained for coin‑heaven scenario.
* **tq\_cratehero** – agent trained for loot‑crate scenario.
* **tq\_allstar** – agent trained on multi‑opponent games.
* **tq\_demonstrator** – **new** helper agent based on `rule_based_agent` that **stores transitions** from every time‑step while it plays. It is used to generate supervised/offline training data for the tabular and DQN pipelines.

### `tq_demonstrator` (data recorder)

Add the following to your agent’s `callbacks.py` to enable per‑time‑step **transition recording**. It creates a unique run folder (e.g. `runs/run_YYYYMMDD_HHMMSS_<id>/`) and appends all game transitions into a compressed pickle stream `transitions_all_games.pkl.gz`.

```python
# callbacks.py (excerpt)
import os
import uuid
import datetime
import gzip
import pickle
from typing import Any, Dict, List

# -------------------------------------------------------------------
# In your callbacks.py
# -------------------------------------------------------------------
def setup_training(self):
    """
    Initialise self for recording transitions.
    """

    # generate unique run name and folder
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = uuid.uuid4().hex[:8]
    self.run_name = f"run_{now}_{short_id}"
    self.run_folder = os.path.join("runs", self.run_name)
    os.makedirs(self.run_folder, exist_ok=True)

    # path for *all* transitions
    self.transitions_file = os.path.join(self.run_folder, "transitions_all_games.pkl.gz")
    # ensure it's fresh
    with gzip.open(self.transitions_file, "wb") as f:
        pass

    # buffer to collect transitions for this round
    self._transitions: List[Dict] = []
    self.game = 1


def game_events_occurred(self,
                         old_game_state: dict,
                         self_action: str,
                         new_game_state: dict,
                         events: List[str]):
    """Record the transition"""
    # RECORD FULL OLD STATE + action + events
    self._transitions.append({
        "state": old_game_state,
        "action": self_action,
        "events": events
    })


def end_of_round(self,
                 last_game_state: dict,
                 last_action: str,
                 events: List[str]):
    """
    At round end: record the final transition, then dump all to a compressed file.
    """
    # record the very final transition
    self._transitions.append({
        "state": last_game_state,
        "action": last_action,
        "events": events
    })

    # -------- save everything into this run's folder --------
    with gzip.open(self.transitions_file, "ab") as f:
        # include game index for easy splitting later
        pickle.dump({"game": self.game, "transitions": self._transitions}, f)

    # clear buffer for next round
    self._transitions.clear()
    self._prev_state = None
    self.game += 1
```

### Exploration logic used for data collection

To diversify the dataset, the demonstrator starts with **random exploration** that gradually decays to a minimum value. This is used when choosing actions to ensure we collect transitions from a wide state‑distribution.

```python
# callbacks.py (excerpt)
from collections import deque
from random import shuffle

import numpy as np

import settings as s
from helpers import get_legal_actions, ACTS


def setup(self):
    self.logger.debug('Successfully entered setup code')
    np.random.seed()
    # Fixed length FIFO queues to avoid repeating the same actions
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    # While this timer is positive, agent will not hunt/attack opponents
    self.ignore_others_timer = 0
    self.current_round       = 0
    self.exploration         = 0.3
    # Drops from 1.0 → ~0.05 over ~100k transitions
    self.exploration_decay   = 1e-6
    self.exploration_min     = 0.1


def reset_self(self):
    self.bomb_history = deque([], 5)
    self.coordinate_history = deque([], 20)
    self.ignore_others_timer = 0


def act(self, game_state):
    """Select the next action with ε‑greedy exploration."""
    # decay ε but keep a minimum
    self.exploration = np.max([self.exploration_min, self.exploration * (1 - self.exploration_decay)])

    # Explore with probability ε
    if np.random.uniform(0, 1) < self.exploration:
        self.logger.info(f'Picking action randomly with exploration {self.exploration}')
        legal_actions = get_legal_actions(game_state=game_state)
        action = np.random.choice(legal_actions)
        return ACTS[action]

    # Otherwise: follow rule set
    self.logger.info('Picking action according to rule set')
    if game_state["round"] != self.current_round:
        reset_self(self)
        self.current_round = game_state["round"]
    # ... compute rule‑based action from here ...
```

---

## Usage

Below are minimal, end‑to‑end scripts to **reproduce the results** shown in the blog post. Adjust paths if your run folders differ.

### Coingrabber

1. **Generate and store raw game data** (dictionary at every time‑step):

```bash
python3 main.py play \
  --agents tq_demonstrator \
  --train 1 --n-rounds 50000 \
  --scenario coin-heaven --no-gui
```

2. **Convert the training data** (tuples of `(states, actions, rewards)`):

```bash
python3 q_build_training_episodes.py agent_code/tq_demonstrator/runs/coin_heaven_50k/
```

3. **Pretrain the agent** on all episodes in 10 batches:

```bash
python3 q_pretrain_tabular.py \
  -i agent_code/tq_demonstrator/runs/coin_heaven_50k/transitions.pkl \
  --training-episodes 5000 --num-chunks 10 \
  -o agent_code/tq_coingrabber/q_table.npz \
  --evaluate --agents tq_coingrabber --scenario coin-heaven
```

4. **Test the agent**:

```bash
python3 main.py play --agents tq_coingrabber --scenario coin-heaven
```

---

### Crate‑hero

1. **Generate and store raw game data**:

```bash
python3 main.py play \
  --agents tq_demonstrator \
  --train 1 --n-rounds 50000 \
  --scenario loot-crate --no-gui
```

2. **Convert the training data**:

```bash
python3 q_build_training_episodes.py agent_code/tq_demonstrator/runs/loot_crate_50k/
```

3. **Pretrain the agent**:

```bash
python3 q_pretrain_tabular.py \
  -i agent_code/tq_demonstrator/runs/loot_crate_50k/transitions.pkl \
  --training-episodes 5000 --num-chunks 10 \
  -o agent_code/tq_cratehero/q_table.npz \
  --evaluate --agents tq_cratehero --scenario loot-crate
```

4. **Test the agent**:

```bash
python3 main.py play --agents tq_cratehero --scenario loot-crate
```

---

### Allstar

1. **Generate and store raw game data** (three opponents: peaceful + two rule‑based):

```bash
python3 main.py play \
  --agents tq_demonstrator peaceful_agent rule_based_agent rule_based_agent \
  --train 1 --n-rounds 50000 --no-gui
```

2. **Convert the training data**:

```bash
python3 q_build_training_episodes.py agent_code/tq_demonstrator/runs/three_rule_based_peaceful_50k/
```

3. **Pretrain the agent** (single dataset):

```bash
python3 q_pretrain_tabular.py \
  --transitions-file agent_code/tq_demonstrator/runs/three_rule_based_peaceful_50k/transitions.pkl \
  -o agent_code/tq_allstar/q_table.npz \
  --evaluate --agents tq_allstar peaceful_agent rule_based_agent rule_based_agent
```

Or mix several datasets (increase `--num-chunks` accordingly):

```bash
python3 q_pretrain_tabular.py \
  --transitions-file \
    agent_code/tq_demonstrator/runs/coin_heaven_50k/transitions.pkl \
    agent_code/tq_demonstrator/runs/loot_crate_50k/transitions.pkl \
    agent_code/tq_demonstrator/runs/three_rule_based_peaceful_50k/transitions.pkl \
  -o agent_code/tq_allstar/q_table.npz \
  --training-episodes 5000 --num-chunks 30 \
  --evaluate --agents tq_allstar peaceful_agent rule_based_agent rule_based_agent
```

4. **Test the agent**:

```bash
python3 main.py play --agents tq_allstar peaceful_agent rule_based_agent rule_based_agent
```

---

## Training / code layout

The Q‑learning training and utilities are split across the following files:

* `q_tabular_agent.py` – **Tabular Q‑learning core** (Q‑table updates, ε‑greedy policy, etc.).
* `q_train_tabular.py` – Training loop for the tabular agent.
* `q_pretrain_tabular.py` – Offline pre‑training from recorded transitions / episodes.
* `q_deep_agent.py` – DQN implementation.
* `q_prioritised_experience_replay.py` – PER buffer used by DQN.
* `q_agent_parent.py` – Shared functionality / base class for Q‑agents.
* `q_build_training_episodes.py` – Converts raw game dictionaries into `(state, action, reward)` tuples.
* `q_optuna.py` – Hyper‑parameter tuning utilities.

---

## Tips & notes

* **Run folders:** `tq_demonstrator` writes to `runs/<auto‑generated>/`. You can symlink or copy to scenario‑named folders (e.g. `coin_heaven_50k/`) to keep things tidy.
* **Decaying exploration:** The demonstrator uses ε‑greedy exploration with decay to enrich the dataset at the start of training.
* **Scenarios:** Use `--scenario coin-heaven` or `--scenario loot-crate` to reproduce the experiments.
* **GPU/CPU:** DQN code will use your default device; check `q_deep_agent.py` for toggles.

---

## License

Add your license here (e.g. MIT).

## Citation

If you use this repository in academic work, please cite it. Add BibTeX here.
