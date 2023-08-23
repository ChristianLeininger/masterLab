import random

class RandomPlayer(object):
    def __init__(self, name):
        super(RandomPlayer, self).__init__()
        self.name = name

    def choose_action(self, state, valid_actions):
        return random.choice(valid_actions)
    
