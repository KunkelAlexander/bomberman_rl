#!/usr/bin/env python3
import argparse
import numpy as np
import pickle
import os

import helpers
import tabular_q_agent

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a TabularQAgent from saved transitions and dump Q-tables."
    )
    # I/O
    p.add_argument(
        "--transitions-file", "-i",
        required=True,
        help="Path to input transitions pickle (list of episodes)."
    )
    p.add_argument(
        "--output-q-file", "-o",
        required=True,
        help="Path to output .npz file where q and q_visits will be saved."
    )

    # hyperparameters
    p.add_argument("--discount",         type=float, default=0.9, help="Discount factor γ")
    p.add_argument(
        "--learning-rate-mode", "-m",
        choices=["fixed","adaptive"], default="adaptive",
        help="Learning rate mode"
    )
    p.add_argument(
        "--learning-rate", "-l",
        type=float, default=1e-1,
        help="Base learning rate α"
    )
    p.add_argument(
        "--learning-rate-decay", "-d",
        type=float, default=0.9999,
        help="Per-update decay multiplier on α (if used)"
    )
    p.add_argument(
        "--initial-q", "-q",
        type=float, default=0.0,
        help="Initial Q-value for unseen state–action pairs"
    )
    p.add_argument(
        "--debug", action="store_true",
        help="Enable debug printing"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Build config dict for the agent
    config = {
        "discount"            : args.discount,
        "learning_rate_mode"  : args.learning_rate_mode,
        "learning_rate"       : args.learning_rate,
        "learning_rate_decay" : args.learning_rate_decay,
        "initial_q"           : args.initial_q,
        "debug"               : args.debug,
        # if your agent_tabular_q needs train_freq or others, add here:
        "train_freq"          : getattr(args, "train_freq", 1),
        "exploration"         : getattr(args, "exploration", 0.0),
        "exploration_decay"   : getattr(args, "exploration_decay", 0.0),
        "exploration_min"     : getattr(args, "exploration_min", 0.0),
    }

    # Instantiate
    agent = tabular_q_agent.TabularQAgent(
        agent_id=0,
        n_actions=helpers.N_ACTIONS,
        n_states=helpers.N_STATES,
        config=config,
    )

    # Load transitions
    if not os.path.isfile(args.transitions_file):
        raise FileNotFoundError(f"Could not find {args.transitions_file}")
    agent.load_transitions(args.transitions_file)

    agent.start_game(is_training=True)
    # Train
    agent.train()

    # Save Q & visits
    out_dir = os.path.dirname(args.output_q_file)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        args.output_q_file,
        q=agent.q,
        q_visits=agent.q_visits
    )
    print(f"Saved Q-table of shape {len(agent.q)} and visit counts to {args.output_q_file}")


if __name__ == "__main__":
    main()
