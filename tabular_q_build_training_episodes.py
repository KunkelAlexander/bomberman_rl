import sys
import gzip
import pickle
from typing import List
from tqdm import tqdm
import os

# adjust these imports if your project structure is different:
from helpers import state_to_features, reward_from_events, ACTS, ACT_BITS, ACTION, DONE, REWARD

def build_episodes_from_transitions(folder: str) -> List[List[List]]:
    """
    Load the single-file gzip of all games and convert into a list of episodes
    for TabularQAgent.training_episodes.
    """
    episodes: List[List[List]] = []

    infile = os.path.join(folder, "transitions_all_games.pkl.gz")

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
                reward = reward_from_events(step.get('events', []))
                done   = (idx == len(transitions) - 1)

                transition = [idx, state, legal_actions, action, reward, done]
                episode.append(transition)



            # Sometimes, the last action will be "None". This happens when the agent dies before choosing an action
            # I believe that the correct way to implement this is to make the state before the terminal state
            if episode[-1][ACTION] == None:
                episode[-2][DONE]    = True
                episode[-2][REWARD] += episode[-1][REWARD]
                episode.pop()


            episodes.append(episode)
            pbar.update(1)

        pbar.close()

    return episodes


def main():
    if len(sys.argv) != 2:
        print("Usage: python build_training_episodes.py "
              "<path/to/transitions>")
        sys.exit(1)

    folder = sys.argv[1]
    episodes = build_episodes_from_transitions(folder)

    # save to disk
    path = os.path.join(folder, "transitions.pkl")
    with open(path, 'wb') as out:
        pickle.dump(episodes, out)

    print(f"Built and saved {len(episodes)} episodes → {path}")

if __name__ == "__main__":
    main()
