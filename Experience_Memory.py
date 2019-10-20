
class Episode_experience():
    def __init__(self):
        self.memory = []

    def add(self, state, action, reward, next_state, done, goal):
        self.memory += [(state, action, reward, next_state, done, goal)]

    def clear(self):
        self.memory = []
