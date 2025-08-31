from q_helpers import state_to_tabular_features, describe_tabular_state
def setup(self):
    pass


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    features = state_to_tabular_features(game_state)
    print(describe_tabular_state(features))
    return game_state['user_input']
