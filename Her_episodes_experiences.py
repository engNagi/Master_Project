# TODO trying different HER strategy using the distance to the object

import numpy as np


class Her_episodes_experiences():
    def __init__(self, buffer_size=1000):
        self.memory = []
        self.buffer_size = buffer_size

    def add(self, state, action, reward, next_state, done, goal):
        if len(self.memory) + 1 >= self.buffer_size:
            self.memory[0:(1+len(self.memory))-self.buffer_size] = []
        self.memory += [(state, action, reward, next_state, done, goal)]

    def clear(self):
        self.memory = []

    def her(self, strategy, her_samples):
        if strategy  == "future":
            #   HER
            for t in range(len(self.memory)):
                for k in range(her_samples):
                    future_samples = np.random.randint(t, len(self.memory))  # index of the future transitiobn
                    # future_samples_idx = ep_experience.memory[t+Her_samples]
                    goal = self.memory[future_samples][3]  # next_state of the future transition
                    goal_pos = goal[-3:]
                    state = self.memory[t][0]
                    action = self.memory[t][1]
                    next_state = self.memory[t][3]
                    next_state_pos = next_state[-3:]
                    done = np.array_equal(next_state_pos, goal_pos)
                    reward = 0 if done else -1
                    self.add(state, action, reward, next_state, done, goal_pos)
        return self.memory


