import numpy as np
import random


class experience_buffer(object):
    def __init__(self, mem_size=1000):
        self.memory = []
        self.mem_size = mem_size

    def add(self, experience):
        if len(self.memory) + len(experience) >= self.mem_size:
            self.memory[0:(len(experience)+len(self.memory)) - self.mem_size] = []
        self.memory.extend(experience)

    def sample(self, batch_size):
        return np.reshape(np.array(random.sample(self.memory, batch_size)), [batch_size, 6])

    def get(self):
        return np.reshape(np.array(self.memory), [len(self.memory), 6])



