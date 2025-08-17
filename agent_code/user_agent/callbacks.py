from q_helpers import state_to_features, describe_state
def setup(self):
    pass


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    features = state_to_features(game_state)
    print(describe_state(features))
    return game_state['user_input']
