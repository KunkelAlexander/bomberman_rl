import sys
import gzip
import pickle
from typing import List
from tqdm import tqdm

# adjust these imports if your project structure is different:
from helpers import state_to_features, reward_from_events, ACTS, ACT_BITS

def build_episodes_from_transitions(infile: str) -> List[List[List]]:
    """
    Load the single-file gzip of all games and convert into a list of episodes
    for TabularQAgent.training_episodes.
    """
    episodes: List[List[List]] = []

    with gzip.open(infile, 'rb') as f:
        # create a progress bar for "games" (unknown total, so leave total=None)
        pbar = tqdm(desc="Loading games", unit="game")
        while True:
            try:
                rec = pickle.load(f)
            except EOFError:
                break

            transitions = rec['transitions']
            episode = []
            for idx, step in enumerate(transitions):
                state = state_to_features(step['state'])
                legal_actions = list(range(len(ACTS)))
                action = ACT_BITS.get(step.get('action'))
                reward = reward_from_events(None, step.get('events', []))
                done   = (idx == len(transitions) - 1)

                transition = [idx, state, legal_actions, action, reward, done]
                episode.append(transition)

            episodes.append(episode)
            pbar.update(1)

        pbar.close()

    return episodes


def main():
    if len(sys.argv) != 3:
        print("Usage: python build_training_episodes.py "
              "<path/to/transitions_all_games.pkl.gz> <output_episodes.pkl>")
        sys.exit(1)

    infile, outfile = sys.argv[1], sys.argv[2]
    episodes = build_episodes_from_transitions(infile)

    # save to disk
    with open(outfile, 'wb') as out:
        pickle.dump(episodes, out)

    print(f"Built and saved {len(episodes)} episodes → {outfile}")

if __name__ == "__main__":
    main()
