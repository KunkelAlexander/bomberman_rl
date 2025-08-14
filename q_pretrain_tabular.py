#!/usr/bin/env python3
import argparse
import numpy as np
import pickle
import subprocess
import os

import helpers
import q_tabular_agent

def parse_args():
    p = argparse.ArgumentParser(
        description="Train a TabularQAgent from saved transitions and dump Q-tables."
    )

    # I/O
    p.add_argument("--transitions-file", "-i", required=True, nargs="+",
                   help="Path(s) to one or more transitions pickle files (list of episodes).")
    p.add_argument("--output-q-file", "-o", required=True,
                   help="Path to output .npz file where q and q_visits will be saved.")

    # Hyperparameters
    p.add_argument("--discount", type=float, default=0.9, help="Discount factor γ")
    p.add_argument("--learning-rate-mode", "-m", choices=["fixed", "adaptive"], default="adaptive")
    p.add_argument("--learning-rate", "-l", type=float, default=1e-1)
    p.add_argument("--learning-rate-decay", "-d", type=float, default=0.9999)
    p.add_argument("--initial-q", "-q", type=float, default=0.7)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--training-transitions", "-tt", type=int,
                       help="Train on a fixed number of transitions per chunk.")
    group.add_argument("--training-episodes", "-te", type=int,
                       help="Train on a fixed number of episodes per chunk.")
    p.add_argument("--num-chunks", type=int, default=10, help="Number of training chunks to run.")
    p.add_argument("--debug", action="store_true")

    # Agent config (additional)
    p.add_argument("--train-freq", type=int, default=1, help="Training frequency.")
    p.add_argument("--exploration", type=float, default=0.0)
    p.add_argument("--exploration-decay", type=float, default=0.0)
    p.add_argument("--exploration-min", type=float, default=0.0)

    # Evaluation options
    p.add_argument("--evaluate", action="store_true",
                   help="Run evaluation after each training chunk.")
    p.add_argument("--agents", nargs="+", default=["rule_based_agent"] * 4,
                   help="Agent names to use during evaluation.")
    p.add_argument("--scenario", default="classic",
                   help="Scenario to use during evaluation.")
    p.add_argument("--main-py", default="main.py",
                   help="Path to main.py script.")
    p.add_argument("--eval-rounds", type=int, default=10,
                   help="Number of rounds for evaluation.")

    return p.parse_args()


def save_snapshot(agent, out_dir, base_name, chunk_idx):
    """Save intermediate .npz and pickle snapshots."""
    np.savez(
        os.path.join(out_dir, base_name + ".npz"),
        q=agent.q,
        q_visits=agent.q_visits
    )

    dicts_dir = os.path.join(out_dir, "dicts")
    os.makedirs(dicts_dir, exist_ok=True)

    with open(os.path.join(dicts_dir, f"q_chunk_{chunk_idx:04d}.pkl"), "wb") as f_q:
        pickle.dump(agent.q, f_q)
    with open(os.path.join(dicts_dir, f"q_visits_chunk_{chunk_idx:04d}.pkl"), "wb") as f_qv:
        pickle.dump(agent.q_visits, f_qv)

    print(f"[Chunk {chunk_idx}] Saved snapshot to {out_dir}")


def evaluate_agent(chunk_idx, out_dir, args):
    """Run main.py play with the trained agent and save stats."""
    dicts_dir = os.path.join(out_dir, "dicts")
    stats_file = os.path.join(dicts_dir, f"eval_chunk_{chunk_idx:04d}.json")

    cmd = [
        "python3", args.main_py,
        "play",
        "--agents", *args.agents,
        "--scenario", args.scenario,
        "--n-rounds", str(args.eval_rounds),
        "--train", "0",
        "--no-gui",
        "--save-stats", stats_file,
        "--match-name", f"chunk_{chunk_idx}"
    ]

    print(f"[Chunk {chunk_idx}] Evaluating agent with {args.eval_rounds} rounds...")
    subprocess.run(cmd, check=True)
    print(f"[Chunk {chunk_idx}] Evaluation stats saved to {stats_file}")


def main():
    args = parse_args()

    config = {
        "discount": args.discount,
        "learning_rate_mode": args.learning_rate_mode,
        "learning_rate": args.learning_rate,
        "learning_rate_decay": args.learning_rate_decay,
        "initial_q": args.initial_q,
        "debug": args.debug,
        "train_freq": args.train_freq,
        "exploration": args.exploration,
        "exploration_decay": args.exploration_decay,
        "exploration_min": args.exploration_min,
    }

    agent = q_tabular_agent.TabularQAgent(
        agent_id=0,
        n_actions=helpers.N_ACTIONS,
        n_states=helpers.N_STATES,
        config=config,
    )

    agent.training_episodes = []
    for tf in args.transitions_file:
        if not os.path.isfile(tf):
            raise FileNotFoundError(f"Could not find {tf}")
        with open(tf, "rb") as f:
            episodes = pickle.load(f)
        agent.training_episodes.extend(episodes)
    print(f"Loaded {len(agent.training_episodes)} episodes from {len(args.transitions_file)} file(s).")

    agent.start_game(is_training=True)

    out_dir = os.path.dirname(args.output_q_file)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.output_q_file))[0]

    chunk_idx = 1
    while chunk_idx <= args.num_chunks:
        print(f"{len(agent.training_episodes)} episodes remaining. "
              f"Visited {len(agent.q)} states")

        train_kwargs = {}
        if args.training_transitions is not None:
            train_kwargs["num_transitions"] = args.training_transitions
        elif args.training_episodes is not None:
            train_kwargs["num_episodes"] = args.training_episodes

        agent.train(**train_kwargs)
        save_snapshot(agent, out_dir, base_name, chunk_idx)

        if args.evaluate:
            evaluate_agent(chunk_idx, out_dir, args)

        chunk_idx += 1


    print("Training complete.")


if __name__ == "__main__":
    main()
