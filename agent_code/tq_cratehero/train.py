import pickle
from typing import List

import events as e
from .callbacks import state_to_features
from .config import ACTIONS, N_ACTIONS, N_STATES, ACTION_STRING_TO_ID
import numpy as np


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.agent.start_game(do_training=True)
    self.iteration = 0
    self.game = 0
    self.cumulative_reward = 0
    self.cumulative_rewards = []


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    reward = reward_from_events(self, events)

    self.cumulative_reward += reward

    # state_to_features is defined in callbacks.py
    self.agent.update(iteration = self.iteration, state = state_to_features(old_game_state), legal_actions = np.arange(N_ACTIONS-1), action = ACTION_STRING_TO_ID[self_action], reward = reward, done = False)
    self.iteration += 1

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    reward = reward_from_events(self, events)

    self.agent.update(iteration = self.iteration, state = state_to_features(last_game_state), legal_actions = np.arange(N_ACTIONS-1), action = ACTION_STRING_TO_ID[last_action], reward = reward, done = True)
    self.agent.final_update(reward = 0) # All the final rewards are handed out before, no additional reward is necessary
    self.agent.train()


    # Store the model
    if self.game % 10 == 0:
        np.savez  (f"q-tables/q-table_{self.game}.pt", q = self.agent.q)
    np.savetxt("cum_rewards.txt", self.cumulative_rewards)

    self.cumulative_reward += reward
    print(f"Cumulative reward: {self.cumulative_reward}")
    self.cumulative_rewards.append(self.cumulative_reward)
    self.cumulative_reward = 0

    self.iteration += 1
    self.game += 1



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED:  0.5,
        e.KILLED_OPPONENT: 1.0,
        e.CRATE_DESTROYED: 0.1,
        e.KILLED_SELF:    -1.0,
        e.SURVIVED_ROUND:  1.0,
        e.GOT_KILLED:     -0.1,
        e.BOMB_DROPPED:    0.02,
        e.MOVED_DOWN:     -0.01,
        e.MOVED_UP:       -0.01,
        e.MOVED_LEFT:     -0.01,
        e.MOVED_RIGHT:    -0.01,
        e.WAITED:         -0.01

    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
