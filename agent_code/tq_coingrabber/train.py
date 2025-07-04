import pickle
from typing import List

import events as e
from .callbacks import state_to_features, reward_from_events
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
    self.agent.start_game(is_training=True)
    self.iteration = 0
    self.game = 0
    self.cumulative_reward = 0
    self.cumulative_rewards = []
    self.exploration_rates  = []
    self.agent.load_transitions(f"transitions/1000_transitions.pickle")


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
    if self.game % 5000 == 0:
        np.savez  (f"q-tables/q-table_{self.game}.pt", q = self.agent.q)
        self.agent.save_transitions(f"transitions/transitions.pickle")

    np.savetxt("cum_rewards.txt", self.cumulative_rewards)
    np.savetxt("exp_rates.txt", self.exploration_rates)

    self.cumulative_reward += reward
    self.cumulative_rewards.append(self.cumulative_reward)
    self.exploration_rates.append(self.agent.exploration)
    self.cumulative_reward = 0

    self.iteration += 1
    self.game += 1


