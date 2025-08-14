# bomberman_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

## Usage


### Coingrabber

- **Generate and store raw game data (game dictionary at every time step)**
    - python3 main.py play --agents tq_demonstrator --train 1 --n-rounds 50000 --scenario coin-heaven --no-gui
- **Convert the training data** (Convert game dictionary tuples of (states, actions, rewards))
    - python3 q_build_training_episodes.py agent_code/tq_demonstrator/runs/coin_heaven_50k/
- **Pretrain the agent**
    - python3 q_pretrain_tabular.py -i agent_code/tq_demonstrator/runs/coin_heaven_50k/transitions.pkl -o agent_code/tq_coingrabber/q_table.npz --evaluate --agents tq_coingrabber --scenario coin-heaven
- **Test the agent**
    - python3 main.py play --agents tq_coingrabber --scenario coin-heaven


### Crate-hero


- **Generate and store raw game data (game dictionary at every time step)**
    - python3 main.py play --agents tq_demonstrator --train 1 --n-rounds 50000 --scenario loot-crate --no-gui
- **Convert the training data** (Convert game dictionary tuples of (states, actions, rewards))
    - python3 q_build_training_episodes.py agent_code/tq_demonstrator/runs/loot_crate_50k/
- **Pretrain the agent**
    - python3 q_pretrain_tabular.py -i agent_code/tq_demonstrator/runs/loot_crate_50k/transitions.pkl -o agent_code/tq_cratehero/q_table.npz --evaluate --agents tq_cratehero --scenario loot-crate
- **Test the agent**
    - python3 main.py play --agents tq_cratehero --scenario loot-crate



### Allstar


- **Generate and store raw game data (game dictionary at every time step)**
    - python3 main.py play --agents tq_demonstrator peaceful_agent rule_based_agent rule_based_agent --train 1 --n-rounds 50000 --no-gui
- **Convert the training data** (Convert game dictionary tuples of (states, actions, rewards))
    - python3 q_build_training_episodes.py agent_code/tq_demonstrator/runs/three_rule_based_peaceful_200k/
- **Pretrain the agent**
    - python3 q_pretrain_tabular.py --transitions-file agent_code/tq_demonstrator/runs/three_rule_based_peaceful_50k/transitions.pkl -o agent_code/tq_allstar/q_table.npz --evaluate --agents tq_allstar peaceful_agent rule_based_agent rule_based_agent
 peaceful_agent rule_based_agent rule_based_agent
    - Or if we want more
    - python3 q_pretrain_tabular.py --transitions-file agent_code/tq_demonstrator/runs/coin_heaven_50k/transitions.pkl agent_code/tq_demonstrator/runs/loot_crate_50k/transitions.pkl agent_code/tq_demonstrator/runs/three_rule_based_peaceful_50k/transitions.pkl -o agent_code/tq_allstar/q_table.npz --evaluate --agents tq_allstar peaceful_agent rule_based_agent rule_based_agent
- **Test the agent**
    - python3 main.py play --agents tq_allstar peaceful_agent rule_based_agent rule_based_agent

