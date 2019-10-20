import numpy as np


class BitFlip():
    def __init__(self, n, reward_type):
        self.n = n  # number of bits
        self.reward_type = reward_type

    def reset(self):
        self.goal = np.random.randint(2, size=self.n)  # a random sequence of 0's and 1's
        self.state = np.random.randint(2, size=self.n)  # another random sequence of 0's and 1's as initial state
        return np.copy(self.state), np.copy(self.goal)

    def step(self, action):
        self.state[action] = 1 - self.state[action]  # flip this bit
        done = np.array_equal(self.state, self.goal)
        if self.reward_type == 'sparse':
            reward = 0 if done else -1
        else:
            reward = -np.sum(np.square(self.state - self.goal))
        return np.copy(self.state), reward, done

    def render(self):
        print("\rstate :", np.array_str(self.state), end=' ' * 10)
